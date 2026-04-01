# Korado, a LINDO-like Linear Optimization Tool

Korado is a CLI tool for linear optimizations that mimics the syntax of LINDO, a popular optimization modeling language. It stands for "**K**oray's **O**perations **R**esearch **A**pp for **D**ecision **O**ptimization". **K**oray is my middle name, and it does sound **cool** in the title, therefore being in it.

The purpose of this tool is to provide a LINDO-like optimization tool for **any** OS. I first started this basic app after I realized that there wasn't a LINDO for MacOS, and the alternative, LINGO, was not user-friendly. I wanted to create a simple, more easily usable and accessible alternative.

## Features

- **LINDO Syntax Compatibility**: Supports the core components of the LINDO modeling language:
    - Objective functions: `MAXIMIZE` (or `MAX`) and `MINIMIZE` (or `MIN`).
    - `SUBJECT TO` (or `ST`, `S.T.`, `SUCH THAT`) section for constraints.
    - `END` statement to conclude the model definition.
    - Comment lines starting with `!`.
    - Constraint labels (e.g., `ROW1)`).
- **Post-END Declarations**: Handles common variable specifications placed after the `END` statement:
    - `FREE`: Unrestricted variables.
    - `GIN`: General integer variables.
    - `INT`: Binary integer variables (0 or 1).
    - `SLB`: Simple Lower Bounds.
    - `SUB`: Simple Upper Bounds.
    - `TITLE`: A title for the model.
- **Interactive Editor**: Provides a simple, full-screen text editor (powered by `curses`) for composing and editing models directly in the terminal.
- **Standard Solver Backend**: Uses the robust and widely-used **PuLP** modeling library with the **CBC (COIN-OR Branch and Cut)** solver.
- **Clear Output**: Presents the solution, including the objective function value, variable values, and binding constraints, in a clean, readable format.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/batukoray/Korado.git
    cd Korado
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install `pulp`, which automatically includes the CBC solver.

## Usage

Run the script from your terminal:

```bash
python3 korado.py
```

This will launch the interactive editor.

### In the Editor

-   **Navigate**: Use the arrow keys (`↑`, `↓`, `←`, `→`), `Home`, and `End` to move the cursor.
-   **Edit**: Type to insert characters. Use `Backspace` and `Delete` to remove them.
-   **New Line**: Press `Enter` to create a new line.
-   **Solve**: Press `Ctrl+G` to solve the model.
-   **Quit**: Press `Ctrl+Q` to exit the program.

### Example

The script includes a built-in example. To run it directly, use the `--example` flag:

```bash
python3 korado.py --example
```

This will immediately solve the following model and print the solution:

```
MIN 50 X1 + 100 X2
ST
7 X1 + 2 X2 >= 28
2 X1 + 12 X2 >= 24
END
```

## How It Works

1.  **Input**: The tool accepts a multi-line string formatted according to LINDO syntax.
2.  **Parsing**: A series of regular expressions and parsing functions break down the input text into its core components:
    -   The optimization sense (`MAX` or `MIN`).
    -   The objective function expression.
    -   A list of constraints, each with a left-hand side, an operator (`<=`, `>=`, `=`), and a right-hand side.
    -   Any post-`END` declarations for variable types and bounds.
3.  **Model Formulation**: The parsed specification is used to build an optimization problem object using the `pulp` library. `LpVariable` objects are created with the appropriate bounds and categories (Continuous, Integer, or Binary).
4.  **Solving**: The problem is passed to the CBC solver.
5.  **Output**: The solver's results (status, objective value, variable values, and constraint slack) are formatted and printed to the console.

## Dependencies

-   **PuLP**: A popular open-source linear programming modeler for Python.
-   **CBC (COIN-OR Branch and Cut)**: A high-performance open-source mixed-integer programming solver that is included with PuLP.

---
*This tool is for educational and simulation purposes and is not an official product of LINDO Systems Inc.*