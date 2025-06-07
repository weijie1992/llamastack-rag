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
