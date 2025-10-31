#!/bin/bash
# Quick script to check CUDA version on the system

echo "=== CUDA Version Check ==="
echo ""

# Check NVIDIA driver and CUDA driver version
echo "1. NVIDIA Driver and CUDA Driver Version:"
nvidia-smi 2>/dev/null | grep "CUDA Version" || echo "  nvidia-smi not available"

echo ""
echo "2. CUDA Toolkit Version (nvcc):"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release" || nvcc --version
else
    echo "  nvcc not found in PATH"
fi

echo ""
echo "3. Check installed CUDA libraries:"
ls -d /usr/local/cuda* 2>/dev/null || echo "  No CUDA installations found in /usr/local/"

echo ""
echo "4. PyTorch CUDA version (if PyTorch is installed):"
python3 -c "import torch; print(f'  PyTorch version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "  PyTorch not installed"

echo ""
echo "=== Recommendation ==="
echo "Based on CUDA driver version:"
echo "  - CUDA 12.1-12.3 -> use cu126"
echo "  - CUDA 12.4-12.6 -> use cu128"  
echo "  - CUDA 12.7+     -> use cu129"
echo ""
echo "For RTX 5090, you likely need cu129 (CUDA 12.9+)"

