# Neural-Machine-Translation-with-Attention

## Usage

To work with this project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/sulaiman-shamasna/Neural-Machine-Translation-with-Attention.git

    ```
    
2. **Set up Python environment:**
    - Ensure you have **Python 3.10.X** or higher installed.
    - Create and activate a virtual environment:
      - For Windows (using Git Bash):
        ```bash
        source env/Scripts/activate
        ```
      - For Linux and macOS:
        ```bash
        source env/bin/activate
        ```

3. **Install dependencies:**
    ```bash
    pip install -r reqs.txt
    ```

4. **Run experiments**
    ```bash
    python run.py
    ```

After the training is done, the trained model will be saved in the directory ```dynamic_translator```. However, for inference, run the following.

5. **Run inference**
    ```bash
    python inference.py
    ```