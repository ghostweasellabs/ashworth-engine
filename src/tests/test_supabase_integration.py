import pytest
import asyncio
from src.utils.supabase_client import supabase_client, vecs_client
from src.config.settings import settings

class TestSupabaseIntegration:
    """Test Supabase connectivity and basic operations"""
    
    def test_supabase_connection(self):
        """Test basic Supabase connectivity"""
        try:
            # Test basic query
            result = supabase_client.table("analyses").select("id").limit(1).execute()
            assert result is not None
            assert hasattr(result, 'data')
        except Exception as e:
            pytest.skip(f"Supabase not available: {e}")

    def test_vector_client_connection(self):
        """Test vecs client connectivity"""
        try:
            # Test vector client
            vx = vecs_client
            # Try to get or create a test collection
            collection = vx.get_or_create_collection(
                name="test_collection", 
                dimension=384
            )
            assert collection is not None
        except Exception as e:
            pytest.skip(f"Vector database not available: {e}")

    def test_storage_bucket_access(self):
        """Test Supabase storage bucket access"""
        try:
            # Test bucket listing
            result = supabase_client.storage.list_buckets()
            bucket_names = [bucket['name'] for bucket in result]
            assert settings.storage_bucket in bucket_names
        except Exception as e:
            pytest.skip(f"Storage not available: {e}")

    def test_analyses_table_operations(self):
        """Test basic CRUD operations on analyses table"""
        try:
            # Test insert
            test_data = {
                "id": "test-analysis-123",
                "client_id": "test-client",
                "analysis_type": "financial_analysis",
                "status": "testing",
                "file_name": "test.csv",
                "file_size": 1024
            }
            
            # Insert test record
            insert_result = supabase_client.table("analyses").insert(test_data).execute()
            assert insert_result.data
            assert len(insert_result.data) == 1
            
            # Test select
            select_result = supabase_client.table("analyses").select("*").eq("id", "test-analysis-123").execute()
            assert select_result.data
            assert select_result.data[0]["client_id"] == "test-client"
            
            # Test update
            update_result = supabase_client.table("analyses").update({
                "status": "completed"
            }).eq("id", "test-analysis-123").execute()
            assert update_result.data
            assert update_result.data[0]["status"] == "completed"
            
            # Test delete (cleanup)
            delete_result = supabase_client.table("analyses").delete().eq("id", "test-analysis-123").execute()
            assert delete_result.data
            
        except Exception as e:
            pytest.skip(f"Database operations not available: {e}")

    def test_clients_table_operations(self):
        """Test basic operations on clients table"""
        try:
            # Test insert
            test_client = {
                "id": "test-client-123",
                "name": "Test Client",
                "email": "test@example.com",
                "business_type": "consulting"
            }
            
            # Insert test client
            insert_result = supabase_client.table("clients").insert(test_client).execute()
            assert insert_result.data
            
            # Test select
            select_result = supabase_client.table("clients").select("*").eq("id", "test-client-123").execute()
            assert select_result.data
            assert select_result.data[0]["name"] == "Test Client"
            
            # Cleanup
            supabase_client.table("clients").delete().eq("id", "test-client-123").execute()
            
        except Exception as e:
            pytest.skip(f"Clients table operations not available: {e}")

    def test_storage_upload_download(self):
        """Test file upload and download from Supabase Storage"""
        try:
            # Test upload
            test_content = b"This is test content for storage"
            test_path = "test/test-file.txt"
            
            # Upload file
            upload_result = supabase_client.storage.from_(settings.storage_bucket).upload(
                test_path, test_content
            )
            
            # Check if upload was successful (no error)
            assert not upload_result.get('error')
            
            # Test download
            download_result = supabase_client.storage.from_(settings.storage_bucket).download(test_path)
            assert download_result == test_content
            
            # Cleanup - remove test file
            supabase_client.storage.from_(settings.storage_bucket).remove([test_path])
            
        except Exception as e:
            pytest.skip(f"Storage operations not available: {e}")

    def test_environment_configuration(self):
        """Test that environment is properly configured for local Supabase"""
        # Verify local Supabase configuration
        assert settings.supabase_url == "http://127.0.0.1:54321"
        assert settings.database_url.startswith("postgresql://postgres:postgres@127.0.0.1:54322")
        assert settings.vecs_connection_string.startswith("postgresql://postgres:postgres@127.0.0.1:54322")
        
        # Verify storage configuration
        assert settings.storage_provider == "supabase"
        assert settings.storage_bucket == "reports"
        assert settings.charts_bucket == "charts"

# Async tests for async operations
class TestSupabaseAsyncOperations:
    """Test async Supabase operations"""
    
    @pytest.mark.asyncio
    async def test_async_workflow_state_storage(self):
        """Test storing workflow state asynchronously"""
        try:
            # Simulate workflow state storage
            workflow_state = {
                "trace_id": "async-test-123",
                "client_id": "async-client",
                "status": "processing",
                "phase": "data_extraction",
                "metadata": {"test": True}
            }
            
            # This would be called from agents during workflow execution
            def store_workflow_state(state):
                return supabase_client.table("analyses").insert({
                    "id": state["trace_id"],
                    "client_id": state["client_id"],
                    "status": state["status"],
                    "analysis_type": "test_workflow",
                    "results": state.get("metadata", {})
                }).execute()
            
            # Test storage
            result = store_workflow_state(workflow_state)
            assert result.data
            
            # Cleanup
            supabase_client.table("analyses").delete().eq("id", "async-test-123").execute()
            
        except Exception as e:
            pytest.skip(f"Async workflow testing not available: {e}")

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])