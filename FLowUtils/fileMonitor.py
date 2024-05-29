import os
import time

class ProtocolFileSystem:
    def __init__(self):
        # Mapping from protocol to a list of paths
        self.protocols = {}

    def register(self, protocol, path):
        # Register a new path under a protocol
        if protocol not in self.protocols:
            self.protocols[protocol] = []
        self.protocols[protocol].append(path)

    def resolve(self, url):
        # Convert a URL with a protocol to a physical path
        protocol, relative_path = url.split('://')
        if protocol in self.protocols:
            for path in self.protocols[protocol]:
                full_path = os.path.join(path, relative_path)
                if os.path.exists(full_path):
                    return full_path
        return None  # Return None if the file does not exist under any registered path

class FileMonitor:
    def __init__(self, file_paths):
        # Initialize with a list of file paths to monitor
        self.file_paths = file_paths
        self.last_modified = {file_path: self.get_last_modified(file_path) for file_path in file_paths}

    def get_last_modified(self, file_path):
        # Return the last modified time of a file
        if os.path.exists(file_path):
            return os.path.getmtime(file_path)
        return None
    
    def checkModified(self, file_path):
        # Check if a file has been modified
        last_modified = self.get_last_modified(file_path)
        if last_modified != self.last_modified[file_path]:
            self.last_modified[file_path] = last_modified
            return True
        return False

    def update_files(self):
        # Check for updates to the files
        updated_files = []
        for file_path in self.file_paths:
            if self.checkModified(file_path):
                updated_files.append(file_path)
        return updated_files

    def add_file(self, file_path):
        # Add a new file to monitor
        if file_path not in self.file_paths:
            self.file_paths.append(file_path)
            self.last_modified[file_path] = self.get_last_modified(file_path)

def test_fileManager():
    # Create the protocol file system and register paths
    pfs = ProtocolFileSystem()
    pfs.register("assets", "C:/document")
    pfs.register("assets", "C:/document/xx")

    # Resolving an asset path
    resolved_path = pfs.resolve("assets://some_file.obj")
    print("Resolved path:", resolved_path)

    # Create a file manager and monitor files
    file_manager = FileMonitor([resolved_path] if resolved_path else [])
    while True:
        updated_files = file_manager.update_files()
        if updated_files:
            print("Updated files:", updated_files)
        time.sleep(1)  # Check every second

if __name__ == '__main__':
    test_fileManager()