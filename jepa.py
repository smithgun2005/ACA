"""JEPA with inverse dynamics model."""

import stable_pretraining as spt
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch import nn

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder=None,
        projector=None,
        pred_proj=None,
        inverse_model=None,
        grounded_coordinate=False,
        grounded_coordinate_scale=1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor



        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.inverse_model = inverse_model



        self.action_residual_fwm = None
        self.self_conditioned_residual_fwm = None
        self.self_conditioned_residual_uses_idm_error = False



        self.latent_flow_residual_closure = None
        self.latent_flow_residual_alpha = 1.0
        self.idm_gap_latent_flow_residual_closure = None
        self.idm_gap_latent_flow_residual_alpha = 1.0








        self.grounded_coordinate = bool(grounded_coordinate)
        self.grounded_coordinate_scale = float(grounded_coordinate_scale)
        image_size = getattr(getattr(self.encoder, "config", None), "image_size", None)
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        elif isinstance(image_size, (list, tuple)) and image_size:
            if len(image_size) == 1:
                self.image_size = (int(image_size[0]), int(image_size[0]))
            else:
                self.image_size = (int(image_size[0]), int(image_size[1]))
        else:
            self.image_size = None

        imagenet_stats = spt.data.dataset_stats.ImageNet
        mean = torch.tensor(imagenet_stats["mean"], dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(imagenet_stats["std"], dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    def _resize_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.dtype == torch.uint8:
            pixels = pixels.to(dtype=torch.float32).div_(255.0)
        else:
            pixels = pixels.float()

        if self.image_size is not None and tuple(pixels.shape[-2:]) != self.image_size:
            pixels = F.interpolate(
                pixels,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return pixels

    def _normalize_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        return (pixels - self.pixel_mean) / self.pixel_std

    def resized_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Resize (b, t, c, h, w) pixels to the encoder's input resolution,
        WITHOUT the ImageNet normalization applied inside encode(). Used by
        grounding losses (e.g. PDC) that need real pixel-space magnitudes.
        """
        b = pixels.size(0)
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        t = pixels.size(1)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        pixels = self._resize_pixels(pixels)
        return rearrange(pixels, "(b t) c h w -> b t c h w", b=b, t=t)

    def _preprocess_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        return self._normalize_pixels(self._resize_pixels(pixels))

    def cls_features(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return pre-projector CLS features for state-only mask scoring."""
        b = pixels.size(0)
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        t = pixels.size(1)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(
            self._preprocess_pixels(pixels), interpolate_pos_encoding=True
        )
        return rearrange(output.last_hidden_state[:, 0], "(b t) d -> b t d", b=b, t=t)

    def encode_masked(self, info, patch_masks: torch.Tensor):
        """Encode images with a fixed patch mask, retaining positional layout.

        ``patch_masks`` is [B,T,N] with one value per ViT image patch.  It is
        allowed to be a straight-through tensor: its hard forward value masks
        pixels, and its soft backward value sends gradients to the policy.
        The placeholder is a single fixed black RGB patch in raw pixel space.
        """
        pixels = info["pixels"]
        b = pixels.size(0)
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        t = pixels.size(1)
        if patch_masks.shape[:2] != (b, t):
            raise ValueError(
                f"patch_masks must start with [B,T]=[{b},{t}], got {tuple(patch_masks.shape)}"
            )
        flat_pixels = rearrange(pixels, "b t ... -> (b t) ...")
        raw_pixels = self._resize_pixels(flat_pixels)
        height, width = raw_pixels.shape[-2:]
        patch_size = getattr(getattr(self.encoder, "config", None), "patch_size", None)
        if isinstance(patch_size, (tuple, list)):
            patch_h, patch_w = map(int, patch_size)
        else:
            patch_h = patch_w = int(patch_size)
        if height % patch_h or width % patch_w:
            raise ValueError(
                f"Encoder input {(height, width)} is not divisible by patch size {(patch_h, patch_w)}"
            )
        grid_h, grid_w = height // patch_h, width // patch_w
        expected_patches = grid_h * grid_w
        if patch_masks.size(-1) != expected_patches:
            raise ValueError(
                f"Mask has {patch_masks.size(-1)} patches; expected {expected_patches}"
            )
        mask_grid = patch_masks.reshape(b, t, grid_h, grid_w)
        pixel_mask = mask_grid.repeat_interleave(patch_h, -2).repeat_interleave(patch_w, -1)
        pixel_mask = rearrange(pixel_mask, "b t h w -> (b t) 1 h w").to(raw_pixels.dtype)



        energy = None
        if self.grounded_coordinate:
            energy = self._pixel_change_energy(
                rearrange(raw_pixels, "(b t) c h w -> b t c h w", b=b)
            ) * self.grounded_coordinate_scale
        masked_pixels = raw_pixels * (1.0 - pixel_mask)
        output = self.encoder(
            self._normalize_pixels(masked_pixels), interpolate_pos_encoding=True
        )
        emb = self.projector(output.last_hidden_state[:, 0])
        emb = rearrange(emb, "(b t) d -> b t d", b=b, t=t)
        if energy is not None:
            emb = torch.cat([emb[..., :-1], energy.to(emb.dtype).unsqueeze(-1)], dim=-1)
        info["emb"] = emb
        if "action" in info:
            info["act_emb"] = (
                self.action_encoder(info["action"])
                if self.action_encoder is not None
                else info["action"]
            )
        return info

    @staticmethod
    def _pixel_change_energy(pixels: torch.Tensor) -> torch.Tensor:
        b, t = pixels.shape[:2]
        if t == 1:
            return pixels.new_zeros(b, 1)
        diff = pixels[:, 1:] - pixels[:, :-1]
        energy = diff.float().pow(2).mean(dim=(2, 3, 4)).sqrt()
        zeros = energy.new_zeros(b, 1)
        return torch.cat([zeros, energy], dim=1)

    def encode(self, info):
        pixels = info["pixels"]
        b = pixels.size(0)
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        pixels = self._resize_pixels(pixels)

        energy = None
        if self.grounded_coordinate:
            energy = self._pixel_change_energy(
                rearrange(pixels, "(b t) c h w -> b t c h w", b=b)
            ) * self.grounded_coordinate_scale

        pixels = self._normalize_pixels(pixels)
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        info["cls_features"] = rearrange(
            pixels_emb, "(b t) d -> b t d", b=b
        )
        emb = self.projector(pixels_emb)
        emb = rearrange(emb, "(b t) d -> b t d", b=b)
        if energy is not None:
            emb = torch.cat([emb[..., :-1], energy.to(emb.dtype).unsqueeze(-1)], dim=-1)
        info["emb"] = emb

        if "action" in info:
            info["act_emb"] = (
                self.action_encoder(info["action"])
                if self.action_encoder is not None
                else info["action"]
            )

        return info

    def predict(self, emb, act_emb, reference_act_emb=None, use_aig=False,
                raw_actions=None, apply_residual=True,
                apply_idm_gap_lfr=True, apply_latent_flow_lfr=True):
        """Predict next state embedding.

        ``reference_act_emb`` is used only by AIG during training.  The
        predictor returns the exact same numerical forward output as the
        usual factual-action call; it merely substitutes the action
        conditioner gradient.  It is omitted in evaluation/planning.
        """
        preds = self.predictor(
            emb, act_emb,
            c_reference=reference_act_emb,
            use_aig=use_aig,
        )
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        if apply_residual and self.self_conditioned_residual_fwm is not None:
            if raw_actions is None:
                raise ValueError(
                    "A self-conditioned residual FWM is attached, so predict() "
                    "needs raw normalized actions via raw_actions=."
                )


            base_delta = (preds - emb).detach()
            residual_condition = raw_actions
            if self.self_conditioned_residual_uses_idm_error:
                if self.inverse_model is None:
                    raise RuntimeError(
                        "SC-ResFWM IDM-error conditioning requires an inverse model"
                    )

                residual_condition = (
                    self.inverse_model(emb.detach(), preds.detach()).detach()
                    - raw_actions
                )
            preds = preds + self.self_conditioned_residual_fwm(
                emb, residual_condition, base_delta
            )
        if apply_residual and self.action_residual_fwm is not None:
            if raw_actions is None:
                raise ValueError(
                    "An Action-Residual FWM is attached, so predict() needs "
                    "raw normalized actions via raw_actions=."
                )
            preds = preds + self.action_residual_fwm(emb, raw_actions, preds)
        if apply_residual and apply_latent_flow_lfr and self.latent_flow_residual_closure is not None:
            preds = preds + float(self.latent_flow_residual_alpha) * self.latent_flow_residual_closure(emb, preds - emb)
        if apply_residual and apply_idm_gap_lfr and self.idm_gap_latent_flow_residual_closure is not None:
            if raw_actions is None:
                raise ValueError("IDM-gap LFR needs raw normalized actions via raw_actions=")
            if self.inverse_model is None:
                raise RuntimeError("IDM-gap LFR requires an inverse dynamics model")



            action_gap = raw_actions - self.inverse_model(emb, preds)
            preds = preds + float(self.idm_gap_latent_flow_residual_alpha) * \
                self.idm_gap_latent_flow_residual_closure(emb, preds - emb, action_gap)
        return preds

    def predict_with_features(self, emb, actions):
        if not hasattr(self.predictor, "forward_with_features"):
            raise TypeError(
                "PC-WM requires predictor.type=ar (AdaLN ARPredictor)"
            )
        act_emb = self.action_encoder(actions) if self.action_encoder is not None else actions
        preds, hidden = self.predictor.forward_with_features(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        if self.self_conditioned_residual_fwm is not None:
            base_delta = (preds - emb).detach()
            residual_condition = actions
            if self.self_conditioned_residual_uses_idm_error:
                if self.inverse_model is None:
                    raise RuntimeError(
                        "SC-ResFWM IDM-error conditioning requires an inverse model"
                    )
                residual_condition = (
                    self.inverse_model(emb.detach(), preds.detach()).detach()
                    - actions
                )
            preds = preds + self.self_conditioned_residual_fwm(
                emb, residual_condition, base_delta
            )
        if self.action_residual_fwm is not None:
            preds = preds + self.action_residual_fwm(emb, actions, preds)
        if self.latent_flow_residual_closure is not None:
            preds = preds + float(self.latent_flow_residual_alpha) * self.latent_flow_residual_closure(emb, preds - emb)
        if self.idm_gap_latent_flow_residual_closure is not None:
            if self.inverse_model is None:
                raise RuntimeError("IDM-gap LFR requires an inverse dynamics model")
            action_gap = actions - self.inverse_model(emb, preds)
            preds = preds + float(self.idm_gap_latent_flow_residual_alpha) * \
                self.idm_gap_latent_flow_residual_closure(emb, preds - emb, action_gap)
        return preds, hidden

    def complete_transition_action(self, z_t, z_next):
        """Recover an action from two endpoints with the shared MTM predictor."""
        if not getattr(self.predictor, "is_masked_transition", False):
            raise TypeError("complete_transition_action requires predictor.type=masked_transition")
        return self.predictor.complete_action(z_t, z_next)

    def complete_endpoint(self, known_emb, act_emb, forward):
        if not getattr(self.predictor, "is_endpoint_completion", False):
            raise TypeError(
                "complete_endpoint requires predictor.type=endpoint_completion"
            )
        preds = self.predictor.complete(known_emb, act_emb, forward=forward)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=known_emb.size(0))

    def predict_with_causal_map(self, emb, actions):
        if not hasattr(self.predictor, "forward_with_causal_map"):
            raise TypeError("Closed-form ACA requires a CAFEPredictor")
        if not isinstance(self.pred_proj, nn.Identity):
            raise RuntimeError("CAFEPredictor must be used without pred_proj")
        return self.predictor.forward_with_causal_map(emb, actions)

    def closed_form_aca(self, z_t, actions, tgt_emb, rho):
        if rho <= 0:
            raise ValueError("CAFE closed-form ACA requires rho > 0")
        pred_emb, C = self.predict_with_causal_map(z_t, actions)
        residual = pred_emb - tgt_emb
        with torch.autocast(device_type=C.device.type, enabled=False):
            C_fp32 = C.float()
            gram = torch.einsum("...dm,...dn->...mn", C_fp32, C_fp32)
            eigvals, eigvecs = torch.linalg.eigh(gram)
        v_min = eigvecs[..., 0].to(actions.dtype)
        Cv = torch.einsum("...dm,...m->...d", C, v_min)
        alignment = (residual * Cv).sum(dim=-1, keepdim=True)

        sign = torch.where(alignment < 0, torch.ones_like(alignment), -torch.ones_like(alignment))
        candidate = actions.detach() + float(rho) * sign * v_min.detach()


        lo = actions.detach().amin(dim=(0, 1), keepdim=True)
        hi = actions.detach().amax(dim=(0, 1), keepdim=True)
        hat_a = candidate.clamp(min=lo, max=hi)
        actual_delta = hat_a - actions.detach()
        pred_emb_hat = self.predict(z_t, hat_a)
        factual_energy = residual.pow(2).sum(dim=-1)
        impostor_energy = (pred_emb_hat - tgt_emb).pow(2).sum(dim=-1)
        margin = actual_delta.pow(2).sum(dim=-1)
        return {
            "pred_emb": pred_emb,
            "pred_loss": residual.pow(2).mean(),
            "pred_loss_hat": (pred_emb_hat - tgt_emb).pow(2).mean(),
            "aca_loss": F.relu(margin + factual_energy - impostor_energy).mean(),
            "aca_margin": margin.mean(),
            "causal_capacity": C.float().pow(2).sum(dim=(-2, -1)).mean(),
            "causal_min_eig": eigvals[..., 0].mean(),
        }

    def predict_action(self, z_t, z_tp1):
        """Predict action from consecutive embeddings using the inverse model."""
        assert self.inverse_model is not None, "No inverse model configured"
        return self.inverse_model(z_t, z_tp1)

    def self_inverting_action_denoising(
        self, z_t, actions, tgt_emb, sigma, step_size=None, create_graph=True
    ):
        sigma = float(sigma)
        if sigma <= 0.0:
            raise ValueError("SI-WM requires loss.si.sigma > 0")
        eta = sigma * sigma if step_size is None else float(step_size)
        if eta <= 0.0:
            raise ValueError("SI-WM requires loss.si.step_size > 0")
        with torch.enable_grad():
            noisy_actions = (
                actions.detach() + sigma * torch.randn_like(actions)
            ).requires_grad_(True)
            action_input = (
                self.action_encoder(noisy_actions)
                if self.action_encoder is not None
                else noisy_actions
            )
            pred_emb = self.predict(z_t.detach(), action_input)
            energy = 0.5 * (pred_emb - tgt_emb.detach()).float().pow(2).sum(dim=-1)
            (score,) = torch.autograd.grad(
                energy.sum(), noisy_actions,
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=True,
            )
            if score is None:
                score = torch.zeros_like(noisy_actions)
            recovered_actions = noisy_actions - eta * score
            loss = (recovered_actions - actions.detach()).float().pow(2).mean()

        return {
            "si_loss": loss,
            "si_energy": energy.detach().mean(),
            "si_score_norm": score.detach().flatten(1).norm(dim=-1).mean(),
            "si_recovery_error": (
                recovered_actions.detach() - actions.detach()
            ).float().pow(2).mean().sqrt(),
        }

    def sample_positive_aca_actions(
        self, z_t, actions, tgt_emb, rho, margin, top_fraction, max_samples,
        action_low, action_high,
    ):
        if rho <= 0 or not 0 < top_fraction <= 1:
            raise ValueError("invalid online ACA sampling hyperparameters")
        mined_actions, hinge = self.mine_aca_actions(
            z_t, actions, tgt_emb, rho=rho, margin=margin,
            action_low=action_low, action_high=action_high,
        )
        positive = torch.nonzero(hinge > 0, as_tuple=False)
        if positive.numel() == 0:
            return [], hinge.new_zeros(())
        values = hinge[positive[:, 0]]
        keep = max(1, int(np.ceil(float(top_fraction) * values.numel())))
        keep = min(keep, int(max_samples))
        best = values.topk(keep).indices
        chosen = []
        for idx in best.tolist():
            batch_idx = positive[idx].item()
            chosen.append((batch_idx, mined_actions[batch_idx].detach()))
        return chosen, values[best].mean()

    def mine_aca_actions(
        self, z_t, actions, tgt_emb, rho, margin, action_low, action_high,
    ):
        if rho <= 0:
            raise ValueError("rho must be positive for ACA action mining")

        action_low = action_low.to(device=actions.device, dtype=actions.dtype)
        action_high = action_high.to(device=actions.device, dtype=actions.dtype)

        with torch.enable_grad():
            action_for_grad = actions.detach().clone().requires_grad_(True)
            action_input = (
                self.action_encoder(action_for_grad)
                if self.action_encoder is not None
                else action_for_grad
            )
            pred_emb = self.predict(z_t.detach(), action_input)
            factual_energy = (pred_emb - tgt_emb.detach()).float().pow(2).mean(dim=(-2, -1))
            (grad_a,) = torch.autograd.grad(
                factual_energy.sum(), action_for_grad, allow_unused=True
            )
        grad_a = torch.zeros_like(action_for_grad) if grad_a is None else grad_a.detach()
        direction = grad_a / grad_a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        mined_actions = (actions.detach() - float(rho) * direction).clamp(
            min=action_low, max=action_high
        )
        mined_input = (
            self.action_encoder(mined_actions)
            if self.action_encoder is not None
            else mined_actions
        )
        with torch.no_grad():
            mined_energy = (
                self.predict(z_t.detach(), mined_input) - tgt_emb.detach()
            ).float().pow(2).mean(dim=(-2, -1))
        hinge = float(margin) + factual_energy.detach() - mined_energy
        return mined_actions, hinge

    def adversarial_action_energy(
        self, z_t, actions, tgt_emb, rho, noise_scale=0.0, margin=0.0,
        encoder_scale="none",
    ):

        with torch.enable_grad():
            actions_for_grad = actions.detach().clone().requires_grad_(True)
            act_emb_for_grad = (
                self.action_encoder(actions_for_grad)
                if self.action_encoder is not None
                else actions_for_grad
            )
            pred_emb = self.predict(z_t, act_emb_for_grad)
            energy = (pred_emb - tgt_emb).pow(2).sum(dim=-1)
            (grad_a,) = torch.autograd.grad(
                energy.sum(), actions_for_grad, retain_graph=True, allow_unused=True
            )
            grad_a = (
                grad_a.detach() if grad_a is not None else torch.zeros_like(actions_for_grad)
            )

        direction = grad_a + float(noise_scale) * torch.randn_like(grad_a)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        lo = actions.detach().amin(dim=(0, 1), keepdim=True)
        hi = actions.detach().amax(dim=(0, 1), keepdim=True)
        hat_a = (actions.detach() - rho * direction).clamp(min=lo, max=hi)

        act_emb_hat = (
            self.action_encoder(hat_a) if self.action_encoder is not None else hat_a
        )
        pred_emb_hat = self.predict(z_t, act_emb_hat)
        pred_loss_hat = (pred_emb_hat - tgt_emb).pow(2).mean()
        pred_loss = energy.mean() / energy.new_tensor(float(pred_emb.size(-1)))
        scale_mode = "none" if encoder_scale is None else str(encoder_scale).lower()
        if scale_mode in ("none", "null", "false", "0", "0.0"):
            aca_loss = F.relu(margin + pred_loss - pred_loss_hat)
            return {
                "pred_emb": pred_emb,
                "pred_loss": pred_loss,
                "pred_loss_hat": pred_loss_hat,
                "aca_loss": aca_loss,
                "aca_loss_pred": aca_loss,
                "aca_loss_encoder": aca_loss,
            }


        act_emb_fact = (
            self.action_encoder(actions.detach())
            if self.action_encoder is not None else actions.detach()
        )
        act_emb_imp = (
            self.action_encoder(hat_a.detach())
            if self.action_encoder is not None else hat_a.detach()
        )
        pred_modules = [
            m for m in (self.predictor, self.action_encoder, self.pred_proj)
            if m is not None
        ]
        req = [[p.requires_grad for p in m.parameters()] for m in pred_modules]
        for m in pred_modules:
            for p in m.parameters():
                p.requires_grad_(False)
        try:
            pred_fact_enc = self.predict(z_t, act_emb_fact)
            pred_imp_enc = self.predict(z_t, act_emb_imp)
            aca_loss_encoder = F.relu(
                margin
                + (pred_fact_enc - tgt_emb).pow(2).mean()
                - (pred_imp_enc - tgt_emb).pow(2).mean()
            )
        finally:
            for m, flags in zip(pred_modules, req):
                for p, flag in zip(m.parameters(), flags):
                    p.requires_grad_(flag)



        pred_fact_pred = self.predict(z_t.detach(), act_emb_fact)
        pred_imp_pred = self.predict(z_t.detach(), act_emb_imp)
        aca_loss_pred = F.relu(
            margin
            + (pred_fact_pred - tgt_emb.detach()).pow(2).mean()
            - (pred_imp_pred - tgt_emb.detach()).pow(2).mean()
        )



        aca_loss = F.relu(margin + pred_loss - pred_loss_hat)
        return {
            "pred_emb": pred_emb,
            "pred_loss": pred_loss,
            "pred_loss_hat": pred_loss_hat,
            "aca_loss": aca_loss,
            "aca_loss_pred": aca_loss_pred,
            "aca_loss_encoder": aca_loss_encoder,
        }

    def action_normal_energy(self, z_t, actions, tgt_emb, create_graph=True):
        with torch.enable_grad():
            action_var = actions.detach().clone().requires_grad_(True)
            action_input = (
                self.action_encoder(action_var)
                if self.action_encoder is not None else action_var
            )
            pred_emb = self.predict(z_t, action_input)
            residual = pred_emb - tgt_emb
            energy = 0.5 * residual.square().sum(dim=-1)
            (grad_action,) = torch.autograd.grad(
                energy.sum(),
                action_var,
                create_graph=bool(create_graph),
                retain_graph=True,
                allow_unused=True,
            )
            if grad_action is None:
                grad_action = torch.zeros_like(action_var)
            loss_an = grad_action.square().sum(dim=-1).mean()
        return {
            "action_normal_loss": loss_an,
            "action_normal_grad_norm": grad_action.detach().square().sum(dim=-1).sqrt().mean(),
        }

    def random_aca_energy(self, z_t, actions, tgt_emb, rho, margin=0.0):
        if rho <= 0:
            raise ValueError("random ACA requires rho > 0")



        direction = torch.randn_like(actions)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        lo = actions.detach().amin(dim=(0, 1), keepdim=True)
        hi = actions.detach().amax(dim=(0, 1), keepdim=True)
        random_actions = (actions.detach() + float(rho) * direction).clamp(
            min=lo, max=hi
        )

        factual_input = (
            self.action_encoder(actions) if self.action_encoder is not None else actions
        )
        random_input = (
            self.action_encoder(random_actions)
            if self.action_encoder is not None else random_actions
        )
        pred_emb = self.predict(z_t, factual_input)
        pred_emb_random = self.predict(z_t, random_input)
        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        pred_loss_random = (pred_emb_random - tgt_emb).pow(2).mean()
        random_aca_loss = F.relu(float(margin) + pred_loss - pred_loss_random)
        return {
            "pred_emb": pred_emb,
            "pred_loss": pred_loss,
            "random_aca_hat_loss": pred_loss_random,
            "random_aca_loss": random_aca_loss,
        }

    def action_energy_gradient_penalty(self, z_t, actions, tgt_emb, rho, create_graph=True):
        if rho <= 0:
            raise ValueError("gradient penalty requires rho > 0")
        with torch.enable_grad():
            action_var = actions.detach().clone().requires_grad_(True)
            action_input = (
                self.action_encoder(action_var)
                if self.action_encoder is not None else action_var
            )
            pred_emb = self.predict(z_t, action_input)
            per_transition_energy = (pred_emb - tgt_emb).pow(2).mean(dim=-1)
            (grad_action,) = torch.autograd.grad(
                per_transition_energy.sum(),
                action_var,
                create_graph=bool(create_graph),
                retain_graph=True,
                allow_unused=True,
            )
            if grad_action is None:
                grad_action = torch.zeros_like(action_var)
            gradient_norm = grad_action.norm(dim=-1)
            gradient_penalty = float(rho) * gradient_norm.mean()
        return {
            "pred_emb": pred_emb,
            "pred_loss": per_transition_energy.mean(),
            "gradient_penalty_loss": gradient_penalty,
            "gradient_penalty_grad_norm": gradient_norm.detach().mean(),
        }

    def counterfactual_only_action_energy(
        self, z_t, actions, tgt_emb, rho, noise_scale=0.0
    ):
        if rho <= 0:
            raise ValueError("counterfactual-only ACA requires rho > 0")
        with torch.enable_grad():
            actions_for_grad = actions.detach().clone().requires_grad_(True)
            factual_input = (
                self.action_encoder(actions_for_grad)
                if self.action_encoder is not None
                else actions_for_grad
            )
            factual_pred = self.predict(z_t.detach(), factual_input)
            factual_energy = (factual_pred - tgt_emb.detach()).pow(2).sum(dim=-1)
            (grad_a,) = torch.autograd.grad(
                factual_energy.sum(), actions_for_grad, allow_unused=True
            )
        grad_a = (
            grad_a.detach() if grad_a is not None else torch.zeros_like(actions_for_grad)
        )
        direction = grad_a
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        lo = actions.detach().amin(dim=(0, 1), keepdim=True)
        hi = actions.detach().amax(dim=(0, 1), keepdim=True)
        hat_a = (actions.detach() - float(rho) * direction).clamp(min=lo, max=hi)
        hat_input = self.action_encoder(hat_a) if self.action_encoder is not None else hat_a
        pred_emb_hat = self.predict(z_t, hat_input)
        counterfactual_loss = (pred_emb_hat - tgt_emb).pow(2).mean()
        factual_energy_mean = factual_energy.mean() / factual_energy.new_tensor(
            float(factual_pred.size(-1))
        )
        return {
            "pred_emb": pred_emb_hat,
            "pred_loss": counterfactual_loss,
            "pred_loss_hat": counterfactual_loss,
            "aca_loss": counterfactual_loss,
            "aca_factual_energy": factual_energy_mean.detach(),
        }

    def bidirectional_adversarial_action_energy(
        self, known_emb, actions, target_emb, forward, rho,
        noise_scale=0.0, margin=0.0,
    ):
        if not getattr(self.predictor, "is_endpoint_completion", False):
            raise TypeError(
                "bidirectional_adversarial_action_energy requires "
                "predictor.type=endpoint_completion"
            )
        with torch.enable_grad():
            actions_for_grad = actions.detach().clone().requires_grad_(True)
            act_emb_for_grad = self.action_encoder(actions_for_grad)
            pred_emb = self.complete_endpoint(
                known_emb, act_emb_for_grad, forward=forward
            )
            energy = (pred_emb - target_emb).pow(2).sum(dim=-1)
            (grad_a,) = torch.autograd.grad(
                energy.sum(), actions_for_grad, retain_graph=True, allow_unused=True
            )
            grad_a = (
                grad_a.detach() if grad_a is not None else torch.zeros_like(actions_for_grad)
            )

        direction = grad_a
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        lo = actions.detach().amin(dim=(0, 1), keepdim=True)
        hi = actions.detach().amax(dim=(0, 1), keepdim=True)
        hat_a = (actions.detach() - float(rho) * direction).clamp(min=lo, max=hi)
        pred_emb_hat = self.complete_endpoint(
            known_emb, self.action_encoder(hat_a), forward=forward
        )
        pred_loss_hat = (pred_emb_hat - target_emb).pow(2).mean()
        pred_loss = energy.mean() / energy.new_tensor(float(pred_emb.size(-1)))
        aca_loss = F.relu(float(margin) + pred_loss - pred_loss_hat)
        return {
            "pred_emb": pred_emb,
            "pred_loss": pred_loss,
            "pred_loss_hat": pred_loss_hat,
            "aca_loss": aca_loss,
        }

    def convergent_aca(
        self, z_t, actions, tgt_emb, rho, max_steps=20, step_size=None,
        grad_tol=1e-5, num_restarts=4, margin=0.0,
    ):
        if rho <= 0:
            raise ValueError("Convergent ACA requires rho > 0")
        B, T, D_act = actions.shape
        center = actions.detach()
        lo = center.amin(dim=(0, 1), keepdim=True)
        hi = center.amax(dim=(0, 1), keepdim=True)
        step_size = float(step_size if step_size is not None else rho / 4)
        R = max(1, int(num_restarts))

        def project(x):
            x = x.clamp(min=lo.unsqueeze(0), max=hi.unsqueeze(0))
            delta = x - center.unsqueeze(0)
            scale = (float(rho) / delta.flatten(2).norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
            x = center.unsqueeze(0) + delta * scale[..., None]
            return x.clamp(min=lo.unsqueeze(0), max=hi.unsqueeze(0))

        with torch.enable_grad():
            seeds = [center]
            for _ in range(R - 1):
                direction = torch.randn_like(center)
                direction = direction / direction.flatten(1).norm(dim=-1, keepdim=True).clamp_min(1e-12).view(B, 1, 1)
                seeds.append(project((center + float(rho) * direction).unsqueeze(0))[0])
            current = torch.stack(seeds).detach()
            trajectory = [current]
            z_search = z_t.detach().unsqueeze(0).expand(R, *z_t.shape).reshape(R * B, *z_t.shape[1:])
            target_search = tgt_emb.detach().unsqueeze(0).expand(R, *tgt_emb.shape).reshape(R * B, *tgt_emb.shape[1:])
            for k in range(int(max_steps)):
                flat = current.reshape(R * B, T, D_act).detach().requires_grad_(True)
                ae = self.action_encoder(flat) if self.action_encoder is not None else flat
                e = (self.predict(z_search, ae) - target_search).pow(2).mean(dim=(-2, -1))
                (grad,) = torch.autograd.grad(e.sum(), flat, allow_unused=True)
                grad = torch.zeros_like(flat) if grad is None else grad.detach()
                grad_norm = grad.flatten(1).norm(dim=-1)
                if grad_norm.max().item() <= float(grad_tol):
                    break
                direction = grad / grad_norm.clamp_min(1e-12).view(R * B, 1, 1)
                eta = step_size / (k + 1) ** .5
                current = project((flat - eta * direction).reshape(R, B, T, D_act)).detach()
                trajectory.append(current)
            candidates = torch.cat(trajectory, dim=0)
            K = candidates.size(0)
            cand_flat = candidates.reshape(K * B, T, D_act)
            z_cand = z_t.detach().unsqueeze(0).expand(K, *z_t.shape).reshape(K * B, *z_t.shape[1:])
            tgt_cand = tgt_emb.detach().unsqueeze(0).expand(K, *tgt_emb.shape).reshape(K * B, *tgt_emb.shape[1:])
            ae = self.action_encoder(cand_flat) if self.action_encoder is not None else cand_flat
            cand_energy = (self.predict(z_cand, ae) - tgt_cand).pow(2).mean(dim=(-2, -1)).reshape(K, B)
            best = cand_energy.argmin(dim=0)
            a_star = candidates[best, torch.arange(B, device=actions.device)].detach()

        act_emb = self.action_encoder(actions) if self.action_encoder is not None else actions
        pred_emb = self.predict(z_t, act_emb)
        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        star_emb = self.action_encoder(a_star) if self.action_encoder is not None else a_star
        pred_loss_hat = (self.predict(z_t, star_emb) - tgt_emb).pow(2).mean()
        aca_loss = F.relu(float(margin) + pred_loss - pred_loss_hat)
        distance = (a_star - center).flatten(1).norm(dim=-1)
        return {
            "pred_emb": pred_emb, "pred_loss": pred_loss,
            "pred_loss_hat": pred_loss_hat, "aca_loss": aca_loss,
            "aca_minimum_distance": distance.mean().detach(),
            "aca_active_ratio": (aca_loss.detach() > 0).float(),
        }

    def planner_consistent_aca(
        self,
        z_t,
        actions,
        tgt_emb,
        rho,
        adversary: str = "random_shooting",
        n_samples: int = 32,
        n_elites: int = 8,
        n_iters: int = 3,
        margin: float = 0.0,
    ):
        B, T, D_act = actions.shape

        with torch.no_grad():
            a_det = actions.detach()
            lo = a_det.amin(dim=(0, 1), keepdim=True)
            hi = a_det.amax(dim=(0, 1), keepdim=True)
            K = n_samples

            z_t_exp = (
                z_t.detach()
                .unsqueeze(1).expand(B, K, *z_t.shape[1:])
                .reshape(B * K, *z_t.shape[1:])
            )
            tgt_exp = (
                tgt_emb.detach()
                .unsqueeze(1).expand(B, K, *tgt_emb.shape[1:])
                .reshape(B * K, *tgt_emb.shape[1:])
            )

            noise = (torch.rand(B, K, T, D_act, device=actions.device) * 2 - 1) * rho
            candidates = (a_det.unsqueeze(1) + noise).clamp(
                lo.unsqueeze(1), hi.unsqueeze(1)
            )

            if adversary == "cem":
                mean = a_det
                std = torch.full_like(mean, rho / 2).clamp_min(1e-4)
                for _ in range(n_iters):
                    noise = torch.randn(B, K, T, D_act, device=actions.device)
                    candidates = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(
                        a_det.unsqueeze(1) - rho, a_det.unsqueeze(1) + rho
                    ).clamp(lo.unsqueeze(1), hi.unsqueeze(1))
                    cand_flat = candidates.reshape(B * K, T, D_act)
                    ae = self.action_encoder(cand_flat) if self.action_encoder is not None else cand_flat
                    ef = (self.predict(z_t_exp, ae) - tgt_exp).pow(2).mean(dim=(-2, -1)).reshape(B, K)
                    topk_idx = ef.topk(n_elites, dim=1, largest=False).indices
                    row = torch.arange(B, device=actions.device).unsqueeze(1).expand(B, n_elites)
                    elites = candidates[row, topk_idx]
                    mean = elites.mean(dim=1)
                    std = elites.std(dim=1).clamp_min(1e-4)

            cand_flat = candidates.reshape(B * K, T, D_act)
            ae_flat = self.action_encoder(cand_flat) if self.action_encoder is not None else cand_flat
            ef = (self.predict(z_t_exp, ae_flat) - tgt_exp).pow(2).mean(dim=(-2, -1)).reshape(B, K)
            best_idx = ef.argmin(dim=1)
            hat_a = candidates[torch.arange(B, device=actions.device), best_idx]

        act_emb = self.action_encoder(actions) if self.action_encoder is not None else actions
        pred_emb = self.predict(z_t, act_emb)
        pred_loss = (pred_emb - tgt_emb).pow(2).mean()

        act_emb_hat = self.action_encoder(hat_a) if self.action_encoder is not None else hat_a
        pred_emb_hat = self.predict(z_t, act_emb_hat)
        pred_loss_hat = (pred_emb_hat - tgt_emb).pow(2).mean()

        aca_loss = F.relu(margin + pred_loss - pred_loss_hat)
        return {
            "pred_emb": pred_emb,
            "pred_loss": pred_loss,
            "pred_loss_hat": pred_loss_hat,
            "aca_loss": aca_loss,
        }

    def rollout(self, info, action_sequence):
        """Rollout the model given an initial info dict and action sequence."""
        assert "pixels" in info, "pixels not in info_dict"
        B, S, T = action_sequence.shape[:3]

        history_size = int(getattr(self.predictor, "num_frames", 1))
        pixel_history = info["pixels"].size(2)

        past_action_history = None
        action_context_stride = 1
        if "action" in info:
            raw_action_dim = info["action"].size(-1)
            token_action_dim = action_sequence.size(-1)
            action_context_stride = token_action_dim // raw_action_dim
            required_action_history = (
                (history_size - 1) * action_context_stride
                if action_context_stride > 1
                else history_size
            )
            if history_size > 1:
                past_action_history = torch.nan_to_num(
                    info["action"][:, :, -required_action_history:, :], 0.0
                )
        elif history_size > 1:
            raise ValueError(
                'info["action"] is required for rollout with predictor history_size > 1'
            )

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init.pop("action", None)
        _init = self.encode(_init)
        if action_context_stride > 1 and history_size > 1:
            required_pixel_history = (history_size - 1) * action_context_stride + 1
            emb_indices = torch.arange(
                pixel_history - required_pixel_history,
                pixel_history,
                action_context_stride,
                device=_init["emb"].device,
            )
            emb = _init["emb"].index_select(1, emb_indices)
        else:
            emb = _init["emb"][:, -history_size:]
        emb = info["emb"] = emb.unsqueeze(1).expand(B, S, -1, -1)

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        action_sequence = torch.nan_to_num(action_sequence, 0.0)
        future_actions = rearrange(action_sequence, "b s ... -> (b s) ...")
        if history_size > 1:
            if action_context_stride > 1:
                past_actions = past_action_history.reshape(
                    B, S, history_size - 1, -1
                )
                past_actions = rearrange(past_actions, "b s ... -> (b s) ...")
            else:
                past_actions = rearrange(
                    past_action_history[:, :, -(history_size - 1) :, :],
                    "b s ... -> (b s) ...",
                )
            actions = torch.cat([past_actions, future_actions], dim=1)
        else:
            actions = future_actions

        for t in range(T):
            emb_context = emb[:, -history_size:]
            act_context = actions[:, t : t + history_size]
            act_emb = (
                self.action_encoder(act_context)
                if self.action_encoder is not None
                else act_context
            )
            pred_emb = self.predict(
                emb_context, act_emb, raw_actions=act_context,
                apply_idm_gap_lfr=(t == 0),
                apply_latent_flow_lfr=(t == 0),
            )[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        return info

    def criterion(self, info_dict: dict):
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))
        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        assert "goal" in info_dict, "goal not in info_dict"
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]
        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_"):]] = goal.pop(k)
        goal.pop("action", None)
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)
        cost = self.criterion(info_dict)
        return cost
