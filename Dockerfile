# syntax=docker/dockerfile:1.4
# 최상단 주석은 작동을 위해 필요하므로 삭제하지 말 것.

ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS dev

# Setup user specific things

ARG GID
ARG UID
ARG GRP
ARG USR
ARG PASSWD=vuno

# Create user with home directory and password-free sudo permissions.
# This may cause security issues. Use at your own risk.
USER root

# useradd -l (--no-log-init) 옵션 추가
# (참고) https://stackoverflow.com/questions/73208471/docker-build-issue-stuck-at-exporting-layers
# (참고) https://stackoverflow.com/questions/48671214/docker-image-size-for-different-user
# (참고) https://github.com/docker/docs/issues/4754
# MS sssd 연동하면서 uid 숫자가 매우 커졌고, exporting image 단계에서 hang 되는 이상 증상이 발생합니다.
# 검색되는 포스트에 따르면 uid 에 따라 image size 가 들쭉날쭉해진다고도 하는데, image size 가 매우 커지면서 hang 되지 않았나 짐작해봅니다.
# 해당 이슈를 회피하는 방법으로 useradd 에 -l 옵션을 추가합니다.
RUN groupadd -f -g ${GID} ${GRP} && \
    useradd --shell $(which zsh) --create-home -u ${UID} -g ${GRP} \
        -p $(openssl passwd -1 ${PASSWD}) ${USR} && \
    echo "${USR} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ARG PROJECT_ROOT=/opt/project
ENV PATH=${PROJECT_ROOT}:/opt/conda/bin:$PATH
ENV PYTHONPATH=${PROJECT_ROOT}

# Python 패키지를 link directory에 추가
RUN echo /opt/conda/lib >> /etc/ld.so.conf.d/conda.conf && ldconfig

USER ${USR}

# 터미널 설정 변경

# Z-shell 관련 패키지: 터미널 UI 및 UX 향상
ARG HOME=/home/${USR}
ARG PURE_PATH=${HOME}/.zsh/pure
ARG ZSH_FILE=${HOME}/.zsh/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
RUN git clone --depth 1 https://github.com/sindresorhus/pure.git ${HOME}/.zsh/pure
RUN git clone --depth 1 https://github.com/zsh-users/zsh-autosuggestions ${HOME}/.zsh/zsh-autosuggestions
RUN git clone --depth 1 https://github.com/zsh-users/zsh-syntax-highlighting.git ${HOME}/.zsh/zsh-syntax-highlighting

RUN {   echo "fpath+=${PURE_PATH}"; \
        echo "autoload -Uz promptinit; promptinit"; \
        echo "prompt pure"; \
        echo "source ${ZSH_FILE}"; \
        echo "alias ll='ls -al'"; \
    } >> ${HOME}/.zshrc

WORKDIR ${PROJECT_ROOT}
CMD ["/bin/zsh"]
