import build123d as bd
import http.server
import socketserver
import threading
import webbrowser
import os
import time

def show(*args, **kwargs):
    """
    Mocked show() function that redirects output to a three-cad-viewer based web page.
    """
    print("\n[Mocked ocp_vscode] show() called.")
    
    # Extract the part
    # User's code: show(tea_cup, names=["tea cup"])
    cad_obj = args[0]
    
    # BuildPart objects usually have a .part attribute which is the Shape
    if hasattr(cad_obj, "part"):
        shape = cad_obj.part
    else:
        shape = cad_obj

    print("Exporting model to GLB for three-cad-viewer...")
    try:
        from build123d import export_gltf
        # Export as a binary GLB for self-contained model
        export_gltf(shape, "model.glb", binary=True)
        print("Success: Generated model.glb")
    except Exception as e:
        print(f"Error exporting GLTF: {e}")
        return

    # Start the server in a separate thread
    threading.Thread(target=run_server, daemon=True).start()
    
    # Wait for server to initialize
    time.sleep(1)
    
    url = "http://localhost:8000"
    print(f"Serving at {url}")
    print("Opening browser with three-cad-viewer...")
    webbrowser.open(url)
    
    print("\nCAD Viewer is active. Press Ctrl+C in this terminal to stop the server.")
    try:
        # Keep the main process alive so the thread continues
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")

def run_server():
    PORT = 8000
    # Current directory should be the sandbox root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    Handler = http.server.SimpleHTTPRequestHandler
    # Allow port reuse to avoid "Address already in use" errors on frequent restarts
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()
