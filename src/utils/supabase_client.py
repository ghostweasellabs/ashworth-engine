from supabase import create_client, Client
import vecs
from src.config.settings import settings

def create_supabase_client() -> Client:
    """Create Supabase client with proper configuration"""
    return create_client(
        supabase_url=settings.supabase_url,
        supabase_key=settings.supabase_service_key
    )

def create_vecs_client() -> vecs.Client:
    """Create vecs client for vector operations"""
    return vecs.create_client(settings.vecs_connection_string)

def get_vector_collection(dimension: int = 1536, collection_name: str = "documents"):
    """Get or create vector collection for RAG"""
    vx = create_vecs_client()
    return vx.get_or_create_collection(
        name=collection_name,
        dimension=dimension
    )

# Initialize clients
supabase_client = create_supabase_client()
vecs_client = create_vecs_client()