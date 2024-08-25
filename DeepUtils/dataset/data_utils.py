import os
from .rawflowReader import read_rootMetaGridresolution,loadOneFlowEntryRawDataSteady,loadOneFlowEntryRawData


def keep_path_last_n_names(path,n):
    """
    Keep only the last two levels of the given path.
    
    :param path: Original path
    :return: Path with only the last two levels
    """
    # Normalize the path to remove any redundant separators or up-level references
    normalized_path = os.path.normpath(path)
    
    # Split the path into parts
    path_parts = normalized_path.split(os.sep)
    
    # Keep only the last two levels
    last_two_levels = os.sep.join(path_parts[-n:])
    last_two_levels=last_two_levels.replace("/","_")
    last_two_levels=last_two_levels.replace("\\","_")
    return last_two_levels

def getDatasetRootaMeta(root_directory):
    return read_rootMetaGridresolution(os.path.join(root_directory, 'meta.json'))