-- Initial data seeding for development

-- Insert a test client
INSERT INTO public.clients (id, name, email, business_type) 
VALUES (
    '550e8400-e29b-41d4-a716-446655440000',
    'Test Client LLC',
    'test@example.com',
    'Small Business'
) ON CONFLICT (id) DO NOTHING;