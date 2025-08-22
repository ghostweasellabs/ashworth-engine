-- Setup Row Level Security (RLS) policies

-- Enable RLS on all tables
ALTER TABLE public.clients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

-- Create policies for clients table
CREATE POLICY "Users can view their own client data" ON public.clients
    FOR SELECT USING (auth.uid()::text = id::text);

CREATE POLICY "Service role can manage all client data" ON public.clients
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- Create policies for analyses table
CREATE POLICY "Users can view analyses for their clients" ON public.analyses
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.clients 
            WHERE id = analyses.client_id 
            AND auth.uid()::text = clients.id::text
        )
    );

CREATE POLICY "Service role can manage all analyses" ON public.analyses
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- Create policies for transactions table
CREATE POLICY "Users can view transactions for their analyses" ON public.transactions
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.analyses a
            JOIN public.clients c ON a.client_id = c.id
            WHERE a.id = transactions.analysis_id 
            AND auth.uid()::text = c.id::text
        )
    );

CREATE POLICY "Service role can manage all transactions" ON public.transactions
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- Create policies for reports table
CREATE POLICY "Users can view reports for their analyses" ON public.reports
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.analyses a
            JOIN public.clients c ON a.client_id = c.id
            WHERE a.id = reports.analysis_id 
            AND auth.uid()::text = c.id::text
        )
    );

CREATE POLICY "Service role can manage all reports" ON public.reports
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- Create policies for documents table (vector store)
CREATE POLICY "Service role can manage all documents" ON public.documents
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');

-- Storage policies
CREATE POLICY "Users can view their own report files" ON storage.objects
    FOR SELECT USING (
        bucket_id IN ('reports', 'charts') 
        AND auth.jwt() ->> 'role' = 'service_role'
    );

CREATE POLICY "Service role can manage all storage objects" ON storage.objects
    FOR ALL USING (auth.jwt() ->> 'role' = 'service_role');