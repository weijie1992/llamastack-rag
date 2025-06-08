# MAKEFILE.MD

## Running the Project

Follow these steps to set up and run the project:

1. **Run the Makefile Target**:
   Use the `runall` target in the `Makefile` to start both Ollama and LlamaStack:
   ```bash
   make runall
2. **Activate the Python Virtual Environment: After the runall target completes, activate the Python virtual environment and install the required dependencies by running:**
    ```bash
    source setup_env.sh
3. **Run the Demo Script: In the same shell session, execute the demo script:**
    ```bash
    python simple_rag.py


## Debugging
database was in `/Users/<phuaweijie>/.llama/distributions/ollama/faiss_store.db `
can load in db client like dbbeaver point to the path 
run query below 
```
SELECT *
FROM kvstore
WHERE key LIKE '%service_requests_db%' ;
```
Should have 2 result `vector_dbs:v3::service_requests_db` `faiss_index:v3::service_requests_db`
`vector_dbs:v3::service_requests_db`
This key stores the vector database configuration and metadata. It contains:
Database configuration (embedding model, dimensions, provider)
Registration information
Database metadata like creation date, settings
Links to associated FAISS index
The v3 indicates the schema version used by Llama Stack.
`faiss_index:v3::service_requests_db`
faiss_index:v3::service_requests_db
This key stores the actual FAISS index data, which includes:
The binary FAISS index (vector embeddings of your documents)
Document-to-vector mappings
FAISS-specific index configuration
Potentially document metadata and content

extension use
-black for formatting

others
cant read .doc seems like need convert all to docx for better ingestion


not working
history not working

where to find run.yaml
- llama-stack/llama_stack/templates/ollama/run.yaml

where to find docker command
- llama-stack/docs/source/distributions/self_hosted_distro/ollama.md

where to find llama client docs
- https://github.com/meta-llama/llama-stack-client-python

issues
- sometimes it will hanged need check `ollama ps` as well as `llama server`
- currently it keep get only other types of result like below dono why it starts with document 4. Need to debugged further, probably is because `samping_params` need to futher understand more the top_p is it really working

```bash
OS fine refers to a document that is issued by the authorities to employers when they are required to pay a fine for non-compliance with regulations or laws.

Document 4 (Score: 0.5745):
This document is used to record information about the employer’s compliance status and any fines imposed on them.

The following fields in this document are relevant:

Employer’s NRIC
Employer’s FIN
Employer’s CPF No.
Employer’s CPF Submission No.
Employer’s NRIC of the company to which the worker belongs to
These fields will be used to record information about the employer’s compliance status and any fines imposed on them.

The following fields are not relevant:

Employer’s NRIC of the worker
Employer’s FIN of the worker
Employer’s CPF No. of the worker
Employer’s CPF Submission No. of the worker
These fields will be used to record information about the worker’s compliance status and any fines imposed on them.

The following field is not relevant:

Employer’s NRIC of the company to which the worker belongs to
This field is only used for employers, while the other fields are used for workers.

Document 5 (Score: 0.5746):
This document is used to record information about the worker’s compliance status and any fines imposed on them.

The following fields in this document are relevant:

Employer’s NRIC
Employer’s FIN
Employer’s CPF No.
Employer’s CPF Submission No.
Employer’s NRIC of the company to which the worker belongs to
These fields will be used to record information about the worker’s compliance status and any fines imposed on them.

The following field is not relevant:

Employer’s CPF No. of the worker
Employer’s CPF Submission No. of the worker
This field is only used for workers, while the other fields are used for employers.

The following field is not relevant:

Employer’s NRIC of the company to which the worker belongs to
This field is only used for employers, while the other fields are used for workers.

Document 6 (Score: 0.5747):
This document is used to record information about the worker’s compliance status and any fines imposed on them.

The following fields in this document are relevant:

Employer’s NRIC
Employer’s FIN
Employer’s CPF No.
Employer’s CPF Submission No.
Employer’s NRIC of the company to which the worker belongs to
These fields will be used to record information about the
