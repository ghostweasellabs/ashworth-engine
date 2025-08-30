"""
Supabase storage manager for report persistence and versioning.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import aiohttp
import aiofiles

from src.config.settings import settings


logger = logging.getLogger(__name__)


class SupabaseStorageManager:
    """
    Manages report storage in Supabase object storage with versioning and metadata.
    """
    
    def __init__(self):
        """Initialize Supabase storage manager."""
        self.supabase_url = settings.supabase_url
        self.service_key = settings.supabase_service_key
        self.anon_key = settings.supabase_anon_key
        
        # Use service key for admin operations, anon key for public access
        self.headers = {
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json",
            "apikey": self.service_key
        }
        
        self.storage_bucket = "reports"
        self.base_url = f"{self.supabase_url}/storage/v1/object"
        self.public_url = f"{self.supabase_url}/storage/v1/object/public/{self.storage_bucket}"
        
        # Initialize bucket if needed
        asyncio.create_task(self._ensure_bucket_exists())
    
    async def _ensure_bucket_exists(self) -> None:
        """Ensure the reports bucket exists in Supabase storage."""
        try:
            async with aiohttp.ClientSession() as session:
                # Check if bucket exists
                bucket_url = f"{self.supabase_url}/storage/v1/bucket/{self.storage_bucket}"
                
                async with session.get(bucket_url, headers=self.headers) as response:
                    if response.status == 404:
                        # Create bucket
                        create_bucket_data = {
                            "id": self.storage_bucket,
                            "name": self.storage_bucket,
                            "public": True,
                            "file_size_limit": 52428800,  # 50MB
                            "allowed_mime_types": [
                                "text/markdown",
                                "application/json",
                                "text/plain",
                                "image/png",
                                "image/svg+xml"
                            ]
                        }
                        
                        create_url = f"{self.supabase_url}/storage/v1/bucket"
                        async with session.post(
                            create_url,
                            headers=self.headers,
                            json=create_bucket_data
                        ) as create_response:
                            if create_response.status == 200:
                                logger.info(f"Created Supabase bucket: {self.storage_bucket}")
                            else:
                                logger.warning(f"Failed to create bucket: {await create_response.text()}")
                    elif response.status == 200:
                        logger.debug(f"Supabase bucket {self.storage_bucket} already exists")
                    else:
                        logger.warning(f"Unexpected response checking bucket: {response.status}")
                        
        except Exception as e:
            logger.warning(f"Failed to ensure bucket exists: {e}")
    
    async def upload_text(
        self,
        content: str,
        file_path: str,
        content_type: str = "text/plain",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Upload text content to Supabase storage.
        
        Args:
            content: Text content to upload
            file_path: Path within the bucket
            content_type: MIME type of the content
            metadata: Optional metadata to store with the file
            
        Returns:
            Public URL of the uploaded file
        """
        try:
            async with aiohttp.ClientSession() as session:
                upload_url = f"{self.base_url}/{self.storage_bucket}/{file_path}"
                
                upload_headers = {
                    "Authorization": f"Bearer {self.service_key}",
                    "Content-Type": content_type,
                    "apikey": self.service_key
                }
                
                # Add metadata headers if provided
                if metadata:
                    for key, value in metadata.items():
                        upload_headers[f"x-metadata-{key}"] = str(value)
                
                async with session.post(
                    upload_url,
                    headers=upload_headers,
                    data=content.encode('utf-8')
                ) as response:
                    if response.status in [200, 201]:
                        public_url = f"{self.public_url}/{file_path}"
                        logger.info(f"Successfully uploaded {file_path} to Supabase")
                        return public_url
                    else:
                        error_text = await response.text()
                        raise Exception(f"Upload failed with status {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"Failed to upload {file_path} to Supabase: {e}")
            raise
    
    async def upload_file(
        self,
        file_path: Union[str, Path],
        storage_path: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Upload a file to Supabase storage.
        
        Args:
            file_path: Local file path to upload
            storage_path: Path within the bucket
            content_type: MIME type (auto-detected if not provided)
            metadata: Optional metadata to store with the file
            
        Returns:
            Public URL of the uploaded file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect content type if not provided
        if not content_type:
            content_type = self._get_content_type(file_path)
        
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            
            async with aiohttp.ClientSession() as session:
                upload_url = f"{self.base_url}/{self.storage_bucket}/{storage_path}"
                
                upload_headers = {
                    "Authorization": f"Bearer {self.service_key}",
                    "Content-Type": content_type,
                    "apikey": self.service_key
                }
                
                # Add metadata headers if provided
                if metadata:
                    for key, value in metadata.items():
                        upload_headers[f"x-metadata-{key}"] = str(value)
                
                async with session.post(
                    upload_url,
                    headers=upload_headers,
                    data=content
                ) as response:
                    if response.status in [200, 201]:
                        public_url = f"{self.public_url}/{storage_path}"
                        logger.info(f"Successfully uploaded {file_path.name} to Supabase")
                        return public_url
                    else:
                        error_text = await response.text()
                        raise Exception(f"Upload failed with status {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"Failed to upload {file_path} to Supabase: {e}")
            raise
    
    async def download_file(
        self,
        storage_path: str,
        local_path: Optional[Union[str, Path]] = None
    ) -> Union[str, bytes]:
        """
        Download a file from Supabase storage.
        
        Args:
            storage_path: Path within the bucket
            local_path: Optional local path to save the file
            
        Returns:
            File content as string or bytes, or local path if saved
        """
        try:
            async with aiohttp.ClientSession() as session:
                download_url = f"{self.public_url}/{storage_path}"
                
                async with session.get(download_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        if local_path:
                            local_path = Path(local_path)
                            local_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            async with aiofiles.open(local_path, 'wb') as f:
                                await f.write(content)
                            
                            logger.info(f"Downloaded {storage_path} to {local_path}")
                            return str(local_path)
                        else:
                            # Try to decode as text, fallback to bytes
                            try:
                                return content.decode('utf-8')
                            except UnicodeDecodeError:
                                return content
                    else:
                        error_text = await response.text()
                        raise Exception(f"Download failed with status {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"Failed to download {storage_path} from Supabase: {e}")
            raise
    
    async def list_files(
        self,
        prefix: str = "",
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List files in the storage bucket.
        
        Args:
            prefix: Filter files by prefix
            limit: Maximum number of files to return
            offset: Number of files to skip
            
        Returns:
            List of file information dictionaries
        """
        try:
            async with aiohttp.ClientSession() as session:
                list_url = f"{self.supabase_url}/storage/v1/object/list/{self.storage_bucket}"
                
                params = {
                    "limit": limit,
                    "offset": offset
                }
                
                if prefix:
                    params["prefix"] = prefix
                
                async with session.post(
                    list_url,
                    headers=self.headers,
                    json=params
                ) as response:
                    if response.status == 200:
                        files = await response.json()
                        logger.debug(f"Listed {len(files)} files with prefix '{prefix}'")
                        return files
                    else:
                        error_text = await response.text()
                        raise Exception(f"List files failed with status {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"Failed to list files with prefix '{prefix}': {e}")
            raise
    
    async def delete_file(self, storage_path: str) -> bool:
        """
        Delete a file from Supabase storage.
        
        Args:
            storage_path: Path within the bucket
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                delete_url = f"{self.base_url}/{self.storage_bucket}/{storage_path}"
                
                async with session.delete(delete_url, headers=self.headers) as response:
                    if response.status == 200:
                        logger.info(f"Successfully deleted {storage_path}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.warning(f"Delete failed with status {response.status}: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to delete {storage_path}: {e}")
            return False
    
    async def get_file_metadata(self, storage_path: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a file in storage.
        
        Args:
            storage_path: Path within the bucket
            
        Returns:
            File metadata dictionary or None if not found
        """
        try:
            async with aiohttp.ClientSession() as session:
                info_url = f"{self.supabase_url}/storage/v1/object/info/{self.storage_bucket}/{storage_path}"
                
                async with session.get(info_url, headers=self.headers) as response:
                    if response.status == 200:
                        metadata = await response.json()
                        return metadata
                    elif response.status == 404:
                        return None
                    else:
                        error_text = await response.text()
                        raise Exception(f"Get metadata failed with status {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"Failed to get metadata for {storage_path}: {e}")
            return None
    
    async def create_signed_url(
        self,
        storage_path: str,
        expires_in: int = 3600
    ) -> str:
        """
        Create a signed URL for temporary access to a file.
        
        Args:
            storage_path: Path within the bucket
            expires_in: URL expiration time in seconds
            
        Returns:
            Signed URL
        """
        try:
            async with aiohttp.ClientSession() as session:
                signed_url_endpoint = f"{self.supabase_url}/storage/v1/object/sign/{self.storage_bucket}/{storage_path}"
                
                data = {"expiresIn": expires_in}
                
                async with session.post(
                    signed_url_endpoint,
                    headers=self.headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        signed_url = f"{self.supabase_url}/storage/v1{result['signedURL']}"
                        return signed_url
                    else:
                        error_text = await response.text()
                        raise Exception(f"Create signed URL failed with status {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"Failed to create signed URL for {storage_path}: {e}")
            raise
    
    def _get_content_type(self, file_path: Path) -> str:
        """
        Get content type based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type string
        """
        extension = file_path.suffix.lower()
        
        content_types = {
            '.md': 'text/markdown',
            '.txt': 'text/plain',
            '.json': 'application/json',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.svg': 'image/svg+xml',
            '.pdf': 'application/pdf',
            '.csv': 'text/csv',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
        
        return content_types.get(extension, 'application/octet-stream')
    
    async def health_check(self) -> bool:
        """
        Check if Supabase storage is accessible.
        
        Returns:
            True if accessible, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                health_url = f"{self.supabase_url}/storage/v1/bucket/{self.storage_bucket}"
                
                async with session.get(health_url, headers=self.headers) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.warning(f"Supabase storage health check failed: {e}")
            return False
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage usage statistics.
        
        Returns:
            Storage statistics dictionary
        """
        try:
            files = await self.list_files(limit=1000)  # Get more files for stats
            
            total_files = len(files)
            total_size = sum(file.get('metadata', {}).get('size', 0) for file in files)
            
            # Group by file type
            file_types = {}
            for file in files:
                name = file.get('name', '')
                extension = Path(name).suffix.lower() if '.' in name else 'no_extension'
                file_types[extension] = file_types.get(extension, 0) + 1
            
            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_types": file_types,
                "bucket_name": self.storage_bucket,
                "base_url": self.public_url
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {
                "error": str(e),
                "total_files": 0,
                "total_size_bytes": 0
            }


# Global storage manager instance
_storage_manager = None


def get_storage_manager() -> SupabaseStorageManager:
    """Get the global storage manager instance."""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = SupabaseStorageManager()
    return _storage_manager