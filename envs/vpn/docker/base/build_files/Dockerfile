# temporary builder image for wireguard tools
# - need coherent system version with base image
# - may need update for properly compiling boringtun
# - 2023-05: `bookworm` is not officially released, stick to `bullseye` for now
FROM debian:bullseye-slim AS builder

RUN apt-get update && apt-get install -y git build-essential curl

# install boringtun userspace implementation of wireguard
# 
# - match versions of debian & boringtun : up to date 0.5.2 compiles ok for bullseye, but could not have it working
#   Continue with 0.4.0
# - glitch: bullseye's apt-get cargo is too old vs boringtun's cargo packages dependencies, need to install
#   up to date rust/cargo
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && . /root/.cargo/env && \
    cargo install --locked --bin boringtun --version ~0.4.0 boringtun

# install needed wireguard-tools
RUN apt-get install -y wireguard-tools

# docker base image for VPN server, Fed-BioMed node and researcher
# - need proper python version for Fed-BioMed
FROM python:3.10-slim-bullseye

# EUCAIM image metadata - pass `--build-arg IMAGE_REVISION=$(git rev-parse HEAD)` to set the revision
ARG IMAGE_REVISION=""
LABEL maintainer="francesco.cremonesi@inria.fr"
LABEL version="6.4.1"
LABEL description="Fed-BioMed FL framework"
LABEL image.source="https://github.com/fedbiomed/fedbiomed"
LABEL image.revision="${IMAGE_REVISION}"
LABEL image.version="0.1"

RUN apt-get update && \
    apt-get install -y iptables iproute2 iputils-ping bash vim net-tools procps build-essential kmod && \
    rm -f /var/log/lastlog /var/log/faillog && \
    ln -s /dev/null /var/log/lastlog && \
    ln -s /dev/null /var/log/faillog

# get wireguard from builder image
COPY --from=builder /root/.cargo/bin/boringtun /usr/bin/
COPY --from=builder /usr/bin/wg* /usr/bin/
