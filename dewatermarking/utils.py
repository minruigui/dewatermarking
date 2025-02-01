import os
def get_unique_filepath(result_path):
    """If the result file already exists, add a numerical suffix before the file extension."""
    base, extension = os.path.splitext(result_path)  # Split the filename and extension
    counter = 1
    
    # Loop until a unique file name is found
    while os.path.exists(result_path):
        result_path = f"{base}_{counter}{extension}"  # Add a numerical suffix before .json
        counter += 1
    
    return result_path
