

def format_filenames(filenames):
    """Changes ex: 20160101.png to 2016-01-01.png"""
    
    formatted_filenames = []
    for filename in filenames:
        name, ext = os.path.splitext(filename)
        if len(name) == 8 and name.isdigit():
            formatted_name = f"{name[:4]}-{name[4:6]}-{name[6:]}"
            formatted_filenames.append(formatted_name + ext)
        else:
            formatted_filenames.append(filename)  # Keep original if format is unexpected
    return formatted_filenames

if __name__ == "__main__":
    import os

    directory = "imageDatasetwithDates/content/NDVI_Data"
    filenames = os.listdir(directory)
    formatted_filenames = format_filenames(filenames)
    
    for original, formatted in zip(filenames, formatted_filenames):
        print(f"Original: {original} -> Formatted: {formatted}")
        os.rename(os.path.join(directory, original), os.path.join(directory, formatted))