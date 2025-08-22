-- Setup analytics tables for financial intelligence
CREATE TABLE public.clients (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name text NOT NULL,
    email text UNIQUE,
    business_type text,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

CREATE TABLE public.analyses (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id uuid REFERENCES public.clients(id) ON DELETE CASCADE,
    analysis_type text NOT NULL,
    status text DEFAULT 'pending',
    file_name text,
    file_size bigint,
    results jsonb DEFAULT '{}',
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

CREATE TABLE public.transactions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id uuid REFERENCES public.analyses(id) ON DELETE CASCADE,
    date date NOT NULL,
    description text NOT NULL,
    amount decimal(15,2) NOT NULL,
    category text,
    tax_category text,
    is_deductible boolean,
    account text,
    currency text DEFAULT 'USD',
    metadata jsonb DEFAULT '{}',
    created_at timestamptz DEFAULT now()
);

CREATE TABLE public.reports (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id uuid REFERENCES public.analyses(id) ON DELETE CASCADE,
    report_type text NOT NULL,
    markdown_content text,
    pdf_path text,
    charts jsonb DEFAULT '[]',
    created_at timestamptz DEFAULT now()
);

-- Create indexes for better query performance
CREATE INDEX ON public.analyses (client_id);
CREATE INDEX ON public.analyses (status);
CREATE INDEX ON public.transactions (analysis_id);
CREATE INDEX ON public.transactions (date);
CREATE INDEX ON public.transactions (category);
CREATE INDEX ON public.reports (analysis_id);

-- Create triggers for updated_at columns
CREATE TRIGGER update_clients_updated_at
    BEFORE UPDATE ON public.clients
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_analyses_updated_at
    BEFORE UPDATE ON public.analyses
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();