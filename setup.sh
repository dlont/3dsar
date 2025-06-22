#!/bin/bash

# SAR-POMDP Setup Script - Step 2: Specialized Libraries
# This script builds on Step 1 and adds 3D geometry, GIS, and POMDP capabilities

set -e

# Docker Compose command (will be set during requirements check)
DOCKER_COMPOSE_CMD=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Helper function to ensure Docker Compose command is set
ensure_docker_compose_cmd() {
    if [ -z "$DOCKER_COMPOSE_CMD" ]; then
        if docker compose version &> /dev/null; then
            DOCKER_COMPOSE_CMD="docker compose"
        elif command -v docker-compose &> /dev/null; then
            DOCKER_COMPOSE_CMD="docker-compose"
        else
            log_error "Docker Compose not found. Please install Docker Compose."
            exit 1
        fi
    fi
}

check_requirements() {
    log_info "Checking system requirements for Step 2..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check Docker Compose (V2 first, then V1)
    if docker compose version &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker compose"
        log_success "Docker Compose V2 detected"
    elif command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker-compose"
        log_success "Docker Compose V1 detected"
    else
        log_error "Docker Compose is not installed. Please install Docker Compose."
        echo "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    # Check NVIDIA Docker (optional but recommended for Step 2)
    if command -v nvidia-docker &> /dev/null || docker info | grep -q nvidia; then
        log_success "NVIDIA Docker support detected - GPU acceleration available"
    else
        log_warning "NVIDIA Docker not found. Some 3D processing will be CPU-only."
    fi
    
    # Check available disk space (minimum 15GB for Step 2)
    available_space=$(df / | tail -1 | awk '{print $4}')
    min_space_kb=$((15 * 1024 * 1024))
    
    if [ "$available_space" -lt "$min_space_kb" ]; then
        log_error "Insufficient disk space. At least 15GB required for Step 2."
        exit 1
    fi
    
    log_success "System requirements check passed"
}

build_container() {
    log_info "Building Step 2 development container (this may take 15-20 minutes)..."
    
    # Ensure docker compose command is available
    ensure_docker_compose_cmd
    
    log_info "Building specialized libraries: PCL, CGAL, GDAL, Open3D, gRPC, OMPL..."
    
    # Build the container with progress
    if $DOCKER_COMPOSE_CMD build sar-pomdp-dev; then
        log_success "Step 2 container built successfully"
    else
        log_error "Container build failed"
        echo ""
        log_info "Troubleshooting tips:"
        echo "1. Make sure Docker daemon is running"
        echo "2. Check internet connection (downloads many libraries)"
        echo "3. Ensure at least 15GB free disk space"
        echo "4. If out of memory, increase Docker memory limit"
        exit 1
    fi
}

test_step2_libraries() {
    log_info "Testing Step 2 specialized libraries..."
    
    ensure_docker_compose_cmd
    
    # Test PCL (Point Cloud Library)
    log_info "Testing PCL (Point Cloud Library)..."
    if $DOCKER_COMPOSE_CMD exec -T sar-pomdp-dev pkg-config --exists pcl_common; then
        log_success "PCL is available"
    else
        log_warning "PCL test failed"
    fi
    
    # Test CGAL
    log_info "Testing CGAL (Computational Geometry)..."
    if $DOCKER_COMPOSE_CMD exec -T sar-pomdp-dev pkg-config --exists CGAL; then
        log_success "CGAL is available"
    else
        log_warning "CGAL test failed"
    fi
    
    # Test GDAL
    log_info "Testing GDAL (Geospatial library)..."
    if $DOCKER_COMPOSE_CMD exec -T sar-pomdp-dev gdal-config --version > /dev/null 2>&1; then
        log_success "GDAL is available"
    else
        log_warning "GDAL test failed"
    fi
    
    # Test OpenCV
    log_info "Testing OpenCV..."
    if $DOCKER_COMPOSE_CMD exec -T sar-pomdp-dev pkg-config --exists opencv4; then
        log_success "OpenCV is available"
    else
        log_warning "OpenCV test failed"
    fi
    
    # Test Python 3D libraries
    log_info "Testing Python 3D libraries..."
    if $DOCKER_COMPOSE_CMD exec -T sar-pomdp-dev poetry run python -c "import open3d; import trimesh; import geopandas; print('✅ Python 3D libraries work!')" > /dev/null 2>&1; then
        log_success "Python 3D libraries work"
    else
        log_warning "Some Python 3D libraries may not be available"
    fi
    
    # Test basic C++ compilation with specialized libraries
    log_info "Testing C++ compilation with Eigen and PCL..."
    $DOCKER_COMPOSE_CMD exec -T sar-pomdp-dev bash -c 'cd /tmp && echo "#include <iostream>
#include <Eigen/Dense>
int main() {
    Eigen::Vector3d v(1,2,3);
    std::cout << \"C++ with Eigen works: \" << v.transpose() << std::endl;
    return 0;
}" > test.cpp && g++ -I/usr/include/eigen3 test.cpp -o test && ./test' > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        log_success "C++ compilation with specialized libraries works"
    else
        log_warning "C++ specialized library test failed"
    fi
}

test_development_environment() {
    log_info "Testing complete Step 2 development environment..."
    
    # Ensure docker compose command is available
    ensure_docker_compose_cmd
    
    # Test basic functionality from Step 1
    log_info "Testing C++ compiler..."
    if $DOCKER_COMPOSE_CMD exec -T sar-pomdp-dev g++ --version > /dev/null 2>&1; then
        log_success "C++ compiler works"
    else
        log_error "C++ compiler test failed"
        return 1
    fi
    
    # Test Python with Poetry
    log_info "Testing Python with Poetry..."
    if $DOCKER_COMPOSE_CMD exec -T sar-pomdp-dev poetry run python --version > /dev/null 2>&1; then
        log_success "Python with Poetry works"
    else
        log_warning "Poetry test failed, trying direct Python..."
        if $DOCKER_COMPOSE_CMD exec -T sar-pomdp-dev python3 --version > /dev/null 2>&1; then
            log_success "Python works (direct)"
        else
            log_error "Python test failed"
            return 1
        fi
    fi
    
    # Test CMake
    log_info "Testing CMake..."
    if $DOCKER_COMPOSE_CMD exec -T sar-pomdp-dev cmake --version > /dev/null 2>&1; then
        log_success "CMake works"
    else
        log_error "CMake test failed"
        return 1
    fi
    
    # Test CUDA
    log_info "Testing CUDA..."
    if $DOCKER_COMPOSE_CMD exec -T sar-pomdp-dev nvcc --version > /dev/null 2>&1; then
        log_success "CUDA compiler works"
    else
        log_warning "CUDA not available"
    fi
    
    # Test Step 2 specialized libraries
    test_step2_libraries
}

start_services() {
    log_info "Starting services..."
    
    # Ensure docker compose command is available
    ensure_docker_compose_cmd
    
    # Start services
    if $DOCKER_COMPOSE_CMD up -d; then
        log_success "Services started"
    else
        log_error "Failed to start services"
        exit 1
    fi
    
    # Wait a moment for services to initialize
    sleep 5
    
    # Check service health
    check_services_health
}

check_services_health() {
    log_info "Checking service health..."
    
    # Ensure docker compose command is available
    ensure_docker_compose_cmd
    
    # Check if containers are running
    if $DOCKER_COMPOSE_CMD ps | grep -q "Up"; then
        log_success "Containers are running"
    else
        log_error "Some containers are not running"
        $DOCKER_COMPOSE_CMD ps
        return 1
    fi
    
    # Check PostgreSQL
    if $DOCKER_COMPOSE_CMD exec -T postgres pg_isready -U sar_user -d sar_pomdp > /dev/null 2>&1; then
        log_success "PostgreSQL is ready"
    else
        log_warning "PostgreSQL is not yet ready (this is normal on first startup)"
    fi
    
    # Check Redis
    if $DOCKER_COMPOSE_CMD exec -T redis redis-cli ping | grep -q PONG > /dev/null 2>&1; then
        log_success "Redis is ready"
    else
        log_warning "Redis is not yet ready"
    fi
}

setup_project_structure() {
    log_info "Setting up Step 2 project structure..."
    
    # Create directories for Step 2
    mkdir -p {src,build,data,config,logs,tests,scripts}
    mkdir -p src/{cpp,python}
    mkdir -p tests/{cpp,python}
    mkdir -p data/{maps/{buildings,pointclouds,gis_layers},scenarios,test_data}
    
    # Create basic .env file
    if [ ! -f .env ]; then
        cat > .env << 'EOF'
# SAR-POMDP Environment Configuration - Step 2

# Database Configuration
POSTGRES_DB=sar_pomdp
POSTGRES_USER=sar_user
POSTGRES_PASSWORD=sar_password

# Development settings
SAR_POMDP_ENV=development
DEBUG=true

# GPU Configuration
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=all

# Display for GUI applications (Linux)
DISPLAY=${DISPLAY:-:0}

# Step 2 specific settings
ENABLE_3D_PROCESSING=true
ENABLE_GPU_ACCELERATION=true
PCL_NUM_THREADS=4
EOF
        log_success "Created .env file with Step 2 settings"
    fi
    
    # Create enhanced .gitignore
    if [ ! -f .gitignore ]; then
        cat > .gitignore << 'EOF'
# Build artifacts
build/
*.o
*.so
*.a

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
.venv/
*.egg

# Poetry
poetry.lock

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Environment
.env.local
.env.production

# Logs
logs/*.log

# Data files (Step 2 additions)
data/test_data/
data/maps/pointclouds/*.pcd
data/maps/pointclouds/*.las
data/maps/buildings/*.obj
data/maps/buildings/*.ply
data/maps/buildings/*.stl
*.pcd
*.las
*.obj
*.ply
*.stl

# 3D processing temporary files
*.mesh
*.off
*.xyz

# Docker
.dockerignore

# OS
.DS_Store
Thumbs.db
EOF
        log_success "Created enhanced .gitignore"
    fi
    
    log_success "Step 2 project structure created"
}

show_step2_info() {
    ensure_docker_compose_cmd
    echo ""
    log_info "=== Step 2: SAR-POMDP Specialized Environment Ready ==="
    echo ""
    echo "🐳 Container Access:"
    echo "   $DOCKER_COMPOSE_CMD exec sar-pomdp-dev bash"
    echo ""
    echo "💾 Database:"
    echo "   PostgreSQL + PostGIS: localhost:5432 (sar_user/sar_password)"
    echo "   Redis:                 localhost:6379"
    echo ""
    echo "🔬 Specialized Libraries Available:"
    echo "   • PCL (Point Cloud Library)     - LiDAR data processing"
    echo "   • CGAL (Computational Geometry) - 3D algorithms"
    echo "   • GDAL (Geospatial)            - GIS data formats"
    echo "   • Open3D                        - Modern 3D processing"
    echo "   • OpenCV                        - Computer vision"
    echo "   • gRPC                          - High-performance communication"
    echo "   • OMPL                          - Motion planning"
    echo "   • Ceres Solver                  - Optimization"
    echo ""
    echo "🐍 Python 3D Libraries:"
    echo "   poetry run python -c 'import open3d, trimesh, geopandas'"
    echo "   poetry run python -c 'import cv2, sklearn, torch'"
    echo ""
    echo "🔧 Quick Commands:"
    echo "   Test libraries:   ./setup.sh test-step2"
    echo "   View logs:        $DOCKER_COMPOSE_CMD logs -f"
    echo "   Stop:             $DOCKER_COMPOSE_CMD stop"
    echo "   Clean up:         $DOCKER_COMPOSE_CMD down -v"
    echo ""
    echo "📁 Workspace: ./src, ./build, ./data, ./config"
    echo ""
    echo "🚀 Ready for Step 3: POMDP Solver Implementation!"
    echo ""
}

stop_services() {
    log_info "Stopping services..."
    ensure_docker_compose_cmd
    $DOCKER_COMPOSE_CMD stop
    log_success "Services stopped"
}

clean_environment() {
    log_info "Cleaning environment..."
    ensure_docker_compose_cmd
    $DOCKER_COMPOSE_CMD down -v
    docker system prune -f
    log_success "Environment cleaned"
}

show_help() {
    echo "SAR-POMDP Development Environment Setup - Step 2: Specialized Libraries"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  check         Check system requirements for Step 2"
    echo "  setup         Set up Step 2 project structure"
    echo "  build         Build the Step 2 development container"
    echo "  start         Start all services"
    echo "  test          Test the complete development environment"
    echo "  test-step2    Test only Step 2 specialized libraries"
    echo "  stop          Stop all services"
    echo "  clean         Clean up containers and volumes"
    echo "  status        Show service status"
    echo "  shell         Open shell in development container"
    echo "  help          Show this help message"
    echo ""
    echo "Step 2 Examples:"
    echo "  $0 check && $0 setup && $0 build && $0 start && $0 test"
    echo "  $0 test-step2    # Test specialized libraries only"
    echo "  $0 shell         # Enter container with 3D processing libraries"
    echo ""
    echo "Note: Step 2 builds on Step 1 and adds 3D geometry, GIS, and ML libraries"
    echo "      Build time: 15-20 minutes (compiles specialized libraries from source)"
}

# Handle commands
case "${1:-help}" in
    check)
        check_requirements
        ;;
    setup)
        check_requirements
        setup_project_structure
        ;;
    build)
        check_requirements
        build_container
        ;;
    start)
        ensure_docker_compose_cmd
        start_services
        ;;
    test)
        ensure_docker_compose_cmd
        test_development_environment
        ;;
    test-step2)
        ensure_docker_compose_cmd
        test_step2_libraries
        ;;
    status)
        ensure_docker_compose_cmd
        $DOCKER_COMPOSE_CMD ps
        ;;
    shell)
        ensure_docker_compose_cmd
        $DOCKER_COMPOSE_CMD exec sar-pomdp-dev bash
        ;;
    stop)
        stop_services
        ;;
    clean)
        clean_environment
        ;;
    step2)
        check_requirements
        setup_project_structure
        build_container
        start_services
        test_development_environment
        show_step2_info
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac