#!/usr/bin/env python3
"""
Test matplotlib display capabilities
"""

import matplotlib
print(f"Default matplotlib backend: {matplotlib.get_backend()}")

# Try different backends that might work with your display
backends_to_try = ['Qt5Agg', 'TkAgg', 'GTK3Agg', 'Qt4Agg']

for backend in backends_to_try:
    try:
        matplotlib.use(backend)
        import matplotlib.pyplot as plt
        print(f"✅ Successfully set backend to: {backend}")
        
        # Test simple plot
        plt.figure(figsize=(8, 6))
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3], 'bo-')
        plt.title('Matplotlib Test Plot')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.grid(True)
        
        print("Attempting to display plot...")
        plt.show()
        print("✅ Plot displayed successfully!")
        break
        
    except Exception as e:
        print(f"❌ Backend {backend} failed: {e}")
        continue
else:
    print("❌ No working display backend found. Using non-interactive backend.")
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    plt.plot([1, 2, 3, 4], [1, 4, 2, 3], 'bo-')
    plt.title('Matplotlib Test Plot (Saved)')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.grid(True)
    
    plt.savefig('matplotlib_test.png', dpi=150, bbox_inches='tight')
    print("✅ Plot saved as matplotlib_test.png")
    plt.close()

print(f"Final backend: {matplotlib.get_backend()}")