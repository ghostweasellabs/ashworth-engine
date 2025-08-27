-- Storage setup for Ashworth Engine reports and uploads

-- Create storage bucket for reports
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'reports',
    'reports',
    false, -- Private bucket
    52428800, -- 50MB limit
    ARRAY['application/pdf', 'text/markdown', 'application/json', 'text/plain']
) ON CONFLICT (id) DO NOTHING;

-- Create storage bucket for uploaded files
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'uploads',
    'uploads',
    false, -- Private bucket
    52428800, -- 50MB limit
    ARRAY[
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', -- .xlsx
        'application/vnd.ms-excel', -- .xls
        'text/csv',
        'application/pdf',
        'image/png',
        'image/jpeg'
    ]
) ON CONFLICT (id) DO NOTHING;

-- Create RLS policies for storage
CREATE POLICY "Allow authenticated users to upload files" ON storage.objects
    FOR INSERT WITH CHECK (bucket_id = 'uploads' AND auth.role() = 'authenticated');

CREATE POLICY "Allow authenticated users to view their uploads" ON storage.objects
    FOR SELECT USING (bucket_id = 'uploads' AND auth.role() = 'authenticated');

CREATE POLICY "Allow authenticated users to delete their uploads" ON storage.objects
    FOR DELETE USING (bucket_id = 'uploads' AND auth.role() = 'authenticated');

CREATE POLICY "Allow authenticated users to access reports" ON storage.objects
    FOR SELECT USING (bucket_id = 'reports' AND auth.role() = 'authenticated');

CREATE POLICY "Allow system to create reports" ON storage.objects
    FOR INSERT WITH CHECK (bucket_id = 'reports');