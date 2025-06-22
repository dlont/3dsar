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

# === MATPLOTLIB GUI BACKEND DEPENDENCIES ===
RUN apt-get update && apt-get install -y \
    # Qt5 dependencies for PyQt5 backend
    qtbase5-dev \
    qttools5-dev \
    qttools5-dev-tools \
    qt5-qmake \
    python3-pyqt5 \
    python3-pyqt5.qtsvg \
    python3-pyqt5.qtopengl \
    libqt5opengl5-dev \
    libqt5svg5-dev \
    # Tkinter dependencies
    python3-tk \
    tk-dev \
    tcl-dev \
    # GTK dependencies (alternative backend)
    libgtk-3-dev \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gtk-3.0 \
    # X11 and display dependencies for GUI
    xvfb \
    x11-utils \
    x11-xserver-utils \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxrandr2 \
    libxss1 \
    libgconf-2-4 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxtst6 \
    libnss3 \
    libcups2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libcairo-gobject2 \
    libgdk-pixbuf2.0-0 \
    # OpenGL dependencies for matplotlib
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Set up virtual display environment variables
ENV DISPLAY=:99
ENV QT_X11_NO_MITSHM=1
ENV XDG_RUNTIME_DIR=/tmp/runtime-root

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

# Create runtime directory for X11
RUN mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root

# Copy Poetry configuration
COPY pyproject.toml /workspace/

# Install Python packages using Poetry
RUN cd /workspace && poetry install

# # Create virtual display startup script
# RUN echo '#!/bin/bash\n\
# # Start virtual display if not already running\n\
# if ! pgrep -x "Xvfb" > /dev/null; then\n\
#     Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &\n\
#     sleep 2\n\
#     echo "Virtual display started on :99"\n\
# fi\n\
# \n\
# # Test matplotlib backends\n\
# python3 -c "\n\
# import matplotlib\n\
# print(f\"Default backend: {matplotlib.get_backend()}\")\n\
# \n\
# # Try Qt5Agg first\n\
# try:\n\
#     matplotlib.use(\"Qt5Agg\")\n\
#     import matplotlib.pyplot as plt\n\
#     fig = plt.figure()\n\
#     plt.close(fig)\n\
#     print(\"✅ Qt5Agg backend working\")\n\
# except Exception as e:\n\
#     print(f\"❌ Qt5Agg failed: {e}\")\n\
#     try:\n\
#         matplotlib.use(\"TkAgg\")\n\
#         import matplotlib.pyplot as plt\n\
#         fig = plt.figure()\n\
#         plt.close(fig)\n\
#         print(\"✅ TkAgg backend working\")\n\
#     except Exception as e2:\n\
#         print(f\"❌ TkAgg failed: {e2}\")\n\
#         matplotlib.use(\"Agg\")\n\
#         print(\"⚠️  Using Agg backend (no GUI)\")\n\
# "\n\
# \n\
# # Execute the command passed to this script\n\
# exec "$@"' > /usr/local/bin/with-display.sh && \
# chmod +x /usr/local/bin/with-display.sh

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

# Default command starts virtual display and bash
CMD ["/bin/bash"]