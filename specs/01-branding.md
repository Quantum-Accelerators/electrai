# Elf-Net: Project Naming & Branding

## Overview

Rebrand from "ElectrAI" / "RHOAR-Net" to **Elf-Net** as an umbrella project name. "Elf" is a backronym (ELF = Electron Localization Function), and the elf character mascot ties the sub-projects together.

## Sub-projects

| Name | Domain | Description |
|------|--------|-------------|
| **Elf-Net** | ML model | The ResNet-based super-resolution model (currently `electrai`) |
| **Elf-Viz** | Visualization | ThreeJS/React webapp for 3D density visualization |
| **Elf-Guess** | Data generation | Initial guess library (Hananeh's GPAW/SAD tooling) |

## Mascot

An elf character with:
- Green-tinted skin (nature/materials science)
- Purple/blue gradient hair (energy/quantum)
- Circuit traces on face/neck (ML/computation)
- Pointed ear, leaves sprouting from hair (natural world)

Two variants created as SVGs:
- `elfnet.svg` — base character, single visible eye, gentle smile, "ELF-NET" text
- `elfvis.svg` — Elvis-themed variant with pompadour, sunglasses, gold accents, "ELF-VIZ" text

## Tasks

- [ ] Finalize character design (iterate on SVGs with team feedback)
- [ ] Decide on repo rename: `electrai` → `elf-net`? Or keep `electrai` as repo name?
- [ ] Update `pyproject.toml` package name, module paths
- [ ] Create GitHub org or update repo description with new branding
- [ ] Add mascot to README, docs, wandb project page
- [ ] Consider: should sub-projects live in same repo (monorepo) or separate repos?

## Open Questions

- Is the team on board with "Elf-Net" vs "RHOAR-Net"? Need buy-in from Andrew/Betsy/Hananeh.
- Should the PyPI package name change? Breaking change for any downstream users.
- Where do the SVGs live long-term? `docs/assets/`? Top-level?
