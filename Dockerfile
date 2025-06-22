# Step 2: Fixed Open3D Build - Ubuntu 24.04 Compatible
# Properly install Open3D dependencies using their script

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install basic system dependencies first
RUN apt-get update && apt-get install -y \
    # Build essentials (from Step 1)
    build-essential \
    # C++ development tools
    g++ \
    clang \
    gfortran \
    ninja-build \
    pkg-config \
    git \
    wget \
    curl \
    unzip \
    # Open3D
    xorg-dev \
    libxcb-shm0 \
    libglu1-mesa-dev \
    # filament linking
    # libc++-dev \
    # libc++abi-dev \
    # libsdl2-dev \
    # libxi-dev \
    # ML
    libtbb-dev \
    # Headless rendering
    libosmesa6-dev \
    # RealSense
    libudev-dev \
    autoconf \
    libtool \
    # SSL/TLS libraries
    libssl-dev \
    libcrypto++-dev \
    # Python development
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    # Basic libraries from Step 1
    libeigen3-dev \
    libboost-all-dev \
    # Essential tools for dependency scripts
    sudo \
    software-properties-common \
    apt-utils \
    # Utilities
    htop \
    vim \
    tree \
    nano \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get remove cmake -y
RUN pip3 install cmake --upgrade

# === STEP 2: Install Open3D with proper dependency handling ===
WORKDIR /tmp

# Clone Open3D and install dependencies using their script
RUN git clone https://github.com/isl-org/Open3D.git
WORKDIR Open3D

RUN ./util/install_deps_ubuntu.sh assume-yes
# Now build Open3D with dependencies properly installed
RUN mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_SHARED_LIBS=OFF \
             -DBUILD_CUDA_MODULE=ON \
             -DBUILD_GUI=OFF \
             -DBUILD_WEBRTC=OFF \
             -DBUILD_TENSORFLOW_OPS=OFF \
             -DBUILD_PYTORCH_OPS=OFF \
             -DBUILD_BENCHMARKS=OFF \
             -DBUILD_EXAMPLES=OFF \
             -DBUILD_UNIT_TESTS=OFF \
             .. \
    && make -j8 \
    && make install \
    && make install-pip-package \
    && make python-package \
    && make pip-package
WORKDIR /tmp

# Install additional Step 2 system packages
RUN apt-get update && apt-get install -y \
    # Core math libraries
    libarmadillo-dev \
    libblas-dev \
    liblapack-dev \
    # Geospatial libraries
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    libgeos-dev \
    # Communication libraries
    libzmq3-dev \
    libprotobuf-dev \
    protobuf-compiler \
    # Image processing
    libfreeimage-dev \
    # 3d
    libcgal-dev \
    # CV
    libopencv-dev \
    # Nonlinear optimization
    libnlopt-dev   \
    # Audio processing
    libasound2-dev \
    libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for development
RUN useradd -m -s /bin/bash developer \
    && usermod -aG sudo developer \
    && echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Configure Poetry to not create virtual environments (so it uses the container's Python)
RUN poetry config virtualenvs.create false

# Set up workspace and permissions
WORKDIR /workspace
# COPY /src /workspace/
# COPY /config /workspace/
# COPY /data /workspace/
# COPY /logs /workspace/
# COPY /tests /workspace/
# RUN mkdir -p /workspace/build && \
#     chown -R developer:developer /workspace

# Copy Poetry configuration
# COPY pyproject.toml /workspace/

# Install Python packages using Poetry

# RUN cd /workspace && \
#     poetry install
# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get autoclean

# Set proper ownership after all installations
RUN chown -R developer:developer /workspace

# Set environment variables for the application
ENV SAR_POMDP_HOME=/workspace
ENV PYTHONPATH=/workspace/src/python:$PYTHONPATH

# Add library paths
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# Security hardening
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Remove unnecessary packages
RUN apt-get autoremove -y \
    && apt-get autoclean \
    && rm -rf /tmp/* /var/tmp/*

# Switch to developer user
USER developer

# Go back to workspace
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]