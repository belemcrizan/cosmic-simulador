# ==================== ESTÁGIO DE CONSTRUÇÃO ====================
FROM python:3.10-slim AS builder

# Configurações de ambiente
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# 1. Instala dependências de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    git \
    wget \
    python3-dev \
    libfftw3-dev \
    libgsl-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Instala dependências Python
RUN pip install --upgrade pip wheel setuptools && \
    pip install numpy cython scipy

# ==================== INSTALAÇÃO DO CLASS ====================
WORKDIR /opt

# 3. Clone e instalação do CLASS com fallback
RUN git clone --depth 1 https://github.com/lesgourg/class_public.git class_public && \
    cd class_public && \
    { git checkout v3.2.0 2>/dev/null || git checkout master; } && \
    make -j$(nproc) && \
    { [ -f libclass.a ] || { echo "Falha na compilação do CLASS!"; exit 1; }; } && \
    cd python && pip install . && \
    python -c "import classy; print('CLASSy instalado com sucesso')"

# ==================== INSTALAÇÃO DO pyExSHalos ====================
WORKDIR /opt

# 4. Clone do repositório
RUN git clone --depth 1 https://github.com/voivodic/ExSHalos.git pyExSHalos && \
    cd pyExSHalos && \
    git rev-parse HEAD > /tmp/commit.hash

# 5. Instalação robusta com verificação alternativa
WORKDIR /opt/pyExSHalos
RUN pip install . || pip install -e . && \
    # Verificação alternativa já que __version__ não está disponível
    python -c "import pyexshalos; print('pyExSHalos importado com sucesso')" || \
    { echo "Falha crítica na instalação do pyExSHalos!"; exit 1; }

# ==================== ESTÁGIO DE RUNTIME ====================
FROM python:3.10-slim

# 6. Dependências de runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgsl27 \
    libfftw3-double3 \
    && rm -rf /var/lib/apt/lists/*

# 7. Copia artefatos do builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /opt/pyExSHalos /opt/pyExSHalos
COPY --from=builder /tmp/commit.hash /opt/pyExSHalos/commit.hash

# 8. Configuração do aplicativo
WORKDIR /app
COPY . .

# 9. Instala dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# 10. Configuração de segurança
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 11. Health check simplificado
HEALTHCHECK --interval=30s --timeout=10s \
    CMD python -c "import pyexshalos, classy"

# 12. Ponto de entrada
ENTRYPOINT ["python", "backend/run_pipeline.py"]