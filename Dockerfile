FROM python:3.9

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools


COPY --chown=user:user requirements.txt /opt/app/
RUN python -m piptools sync requirements.txt

COPY --chown=user:user src /opt/app/src
COPY --chown=user:user pretrained_models /opt/app/pretrained_models

ENTRYPOINT [ "python", "-m", "src.process" ]