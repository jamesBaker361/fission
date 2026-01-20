import objaverse
import objaverse.xl as oxl

OBJAVERSE_DIR="objaverse"

annotations = oxl.get_annotations(
    download_dir=OBJAVERSE_DIR # default download directory
)

alignment_annotations = oxl.get_alignment_annotations(
    download_dir=OBJAVERSE_DIR# default download directory
)

sampled_df=annotations.head()



from typing import Any, Dict, Hashable

def handle_found_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any]
) -> None:
    print("\n\n\n---HANDLE_FOUND_OBJECT CALLED---\n",
          f"  {local_path=}\n  {file_identifier=}\n  {sha256=}\n  {metadata=}\n\n\n")
    
    
oxl.download_objects(objects=sampled_df,download_dir=OBJAVERSE_DIR,handle_found_object=handle_found_object)