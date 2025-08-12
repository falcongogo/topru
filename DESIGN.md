# Design Document

## Architecture

This application is composed of the following modules:

- **`app.py`**:
  - Responsible for building the UI and handling user interaction using Streamlit.
  - It accepts user inputs for scores, dealer (oya), bonus sticks (tsumibo), and riichi sticks (kyotaku).
  - It calls `calculate_conditions` to perform the calculation and formats the results for display.

- **`calculate_conditions.py`**:
  - The core module responsible for the win condition calculation logic.
  - It calculates the initial point difference to the top player.
  - It then computes the required points for three scenarios: "Direct Ron," "Ron from others," and "Tsumo."
  - It passes the required points for each scenario to `points_lookup.py` to get the corresponding hand (fu/han or Mangan rank).

- **`points_lookup.py`**:
  - A reverse-lookup module that finds the smallest hand to achieve a given score.
  - It searches for hands below Mangan (by fu/han) based on the tables in `points_table.py`.
  - It also contains the logic for hands of Mangan or higher (Mangan, Haneman, Baiman, etc.) and returns the appropriate rank.

- **`points_table.py`**:
  - A data module that defines the points table for non-Mangan hands based on "fu" and "han".
  - It holds point information for four patterns: child/parent and ron/tsumo.

## Point Calculation Specification

The point calculation is executed as follows:

1.  **Calculate Point Difference to Top**:
    - The system calculates the basic point difference required for a win: `(Top Player's Score - Your Score + 1)`.

2.  **Calculate Required Score per Scenario**:
    - Factoring in **bonus sticks (tsumibo)** and **riichi sticks (kyotaku)**, the system calculates the points needed to win in each scenario.
    - **Direct Ron**: Winning directly from the top player means the point swing is doubled (i.e., the required raw score is halved).
    - **Ron from Others**: Winning from a player other than the top player requires covering the simple point difference.
    - **Tsumo**: The bonus points from tsumibo are calculated differently than for a ron win (+400 points per stick). The payout ratios for parent (oya) vs. child (ko) are handled by separate logic.

3.  **Reverse Lookup of Hand (`reverse_lookup`)**:
    - `points_lookup.py` searches for the minimum hand that satisfies the required score for each scenario.
    - It first checks if the score is Mangan-level or higher. If so, it returns the rank ("Mangan" to "Yakuman").
    - For scores below Mangan, it consults `points_table.py` to find the smallest "fu" and "han" combination that exceeds the required score.
    - If no such hand exists, it is marked as "Impossible."

4.  **Display Results**:
    - `app.py` takes the calculated results for each scenario (rank, display points, total points gained) and displays them in a user-friendly card format.
