# Jetson

Jetson is a family of microcontrollers made by Nvidia - seemingly in direct
competition with the Raspberry Pi, which is what makes it so confusing that
Waleed has mentioned both.

They use a Linux-based operating system, called Linux for Tegra (Tegra being the
system on a chip processor that Jetsons use).

The Nano has a CUDA enabled GPU, an ARM-based CPU, 4 GB RAM and 16 GB disk
space, so it's definitely meant to be a beafy microcontroller, not one of those
simple ATmega328P's. It has a built-in camera (one of the things we really
like). It seems built for AI applications like ours, as well as IoT. Advertises
itself as available for $99 (likely USD)

Fitting its name, the Nano is the low-end of the Jetson family of products.

The JetPack SDK, the set of tools that come with a Jetson, features accelerated
libraries for things like machine learning and graphics - so it seems like a
good fit for our application - or, advertises itself as such.

[Website](https://developer.nvidia.com/embedded/jetson-nano)

## Using Jetson (Talk Waleed gave us to watch)

Alright, so there's not that much useful information in this video, not entirely
sure why Waleed asked us to watch this video...

### D3 Engineering

- Electronics, firmware, and systems design. Do a lot of embedded systems.
- Focuses on autonomous systems, edge ai, and robotics, safety, and control.

### What makes an Edge AI system

- start with a compute engine (brain).
- add some sensors.
- sprinkle in some cloud connectivity.

### Jetson

Jetson are (or seem to be) a set of microcontrollers. They all work with the
same software base.

### Case study: smart entry system

#### High-level requirements

- contactless system, allow/deny access to individuals entering controlled area.
- multi camera angles
- real-time updates to central control center
- scalable and easy to field update
- covid features: temp check, social distance, mask wearing
