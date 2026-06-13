# Server-aided password authentication using Module-LWE

## Which file should I run?

> **For Google Colab, run `main.ipynb`.**

That is the only file you need to upload to Colab. Do not run `test_main.py`, and
you do not need to upload `main.py` when you use the notebook.

### Important: do not copy red and green lines

If GitHub shows **red lines and green lines**, you are looking at the pull
request's **Files changed** (diff) page:

- Red lines are old lines that were removed.
- Green lines are new lines that were added.
- The diff is only for reviewing changes. It is **not** a runnable notebook or
  Python file.

**Do not copy either color into Colab.** Instead, download the complete
`main.ipynb` file:

1. Open the repository's **Code** tab—not the pull request's **Files changed**
   tab.
2. Click `main.ipynb`.
3. Click **Download raw file** (the download icon near the top-right of the
   file).
4. Upload that downloaded `main.ipynb` to Google Colab.

The downloaded notebook already contains the complete, current program. There
is nothing to merge, delete, or paste manually.

| File | Purpose | Run in Colab? |
|---|---|---|
| **`main.ipynb`** | Ready-to-run Colab notebook and recommended entry point | **Yes** |
| `main.py` | Same program as a normal Python script | Optional; not needed with the notebook |
| `test_main.py` | Automated developer tests | No |
| `README.md` | Documentation | No |

Choose exactly one entry point:

- **Google Colab:** download and run the complete `main.ipynb`.
- **Local terminal:** download and run the complete `main.py`.

Never combine `main.ipynb` and `main.py`, and never copy code from a red/green
diff view.

## Exact Google Colab steps

1. Download **`main.ipynb`** from this repository to your computer.
2. Open [Google Colab](https://colab.research.google.com/).
3. In Colab, select **File → Upload notebook**.
4. Select the downloaded **`main.ipynb`** file.
5. After it opens, select **Runtime → Run all**.
   - Alternatively, click the ▶ button beside the large Python code cell.
6. Wait for the program to display `Please register at least 3 users to begin.`
7. Enter a username and password in each input box Colab displays.
8. After the initial registrations, use the numbered menu to register,
   authenticate, rotate the server seed, or exit.
9. The latency and peak-memory graphs appear directly below the running cell.

NumPy and Matplotlib are already available in standard Colab runtimes, so no
installation cell is required.

## Optional: run `main.py` in Colab instead

Use this only if you specifically prefer a `.py` file. Upload `main.py` through
Colab's Files panel and then run this in a notebook cell:

```python
%run main.py
```

Do not run both `main.ipynb` and `main.py`; they contain the same application.

## What the prototype implements

The prototype uses the ML-KEM-512 ring parameters `n = 256`, `q = 3329`, and
module rank `l = 2`. Its lattice layer implements the FIPS 203 forward NTT and
quadratic base multiplication, SHAKE-128 seed expansion, and `CBD_eta` sampling
with `eta = 3` for independently derived `s_i`, `e_i`, and `h_i` vectors.

The public matrix `A` is sampled from SHAKE-128-expanded CSPRNG seed material.
Sampling repeats until its determinant is a unit of
`Z_q[X]/(X^256 + 1)`. The implementation computes and verifies `A_inverse` at
startup. Master seeds and per-user salts are generated with
`secrets.token_bytes(32)`, and hash/XOF inputs use length-prefixing and domain
separation.

Registration, authentication, and key rotation report latency and peak Python
allocation. Latency is measured with both `tracemalloc` and cyclic garbage collection
disabled, while peak memory is collected in a separate probe; memory tracing
therefore does not inflate or destabilize the latency samples. The code performs
an unmeasured warm-up calculation before collecting user-facing timings.

Each graph point uses the median of three independently prepared latency runs to
reduce one-off Colab scheduling noise. Registration and authentication graphs
show total batch latency. Because one key-rotation command processes every
stored account, key-rotation graphs and summary rows report **latency and memory
per updated user**. The program also prints and stores total batch rotation time
separately. In the exit summary, `Key Rotation / User` is the requested average
per updated credential record, while `Key Rotation / Batch` is the complete time
for one menu-driven rotation command. The summary also prints separate counts
for rotation commands and user records processed.

The startup banner prints build identifier `2026.06-per-user-rotation-v2`. If
Colab does not show this identifier, it is running an older cached/uploaded copy
of `main.ipynb`; upload the latest notebook and use **Runtime → Restart session
and run all**.

## Local execution

Install the dependencies and run the Python script:

```bash
python -m pip install numpy matplotlib
python main.py
```

## Developer tests

```bash
MPLBACKEND=Agg python -m unittest -v
```

## Security scope

The arithmetic and parameter choices align with the stated ML-KEM-512
Module-LWE target, but this repository implements a password protocol rather
than ML-KEM itself. It has not received independent cryptographic review and
must not be treated as a production credential-storage system solely because it
uses standardized lattice parameters.
