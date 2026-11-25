# ferricast
[WIP] centralize cast protocols into one app

## GPU selection

Hardware encoders and decoders run on the GPU `FERRICAST_GPU` (or
`ferricast-gui --gpu`) names: `auto` (the GPU that renders the screen,
else a discrete one), `same-as-capture`, `integrated`, `discrete`, a render
node (`/dev/dri/renderD129`) or a PCI address (`0000:01:00.0`).
`ferricast-gui --list-gpus` lists the GPUs and the hardware codecs on each.
