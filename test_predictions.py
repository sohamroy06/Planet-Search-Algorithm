"""Quick test of small-planet predictions after retraining."""
import requests

URL = "http://127.0.0.1:5000/api/predict"

tests = [
    ("TRAPPIST-1e (0.92 Re)", {
        "koi_period": 6.1, "koi_depth": 700, "koi_duration": 0.85,
        "koi_steff": 2566, "koi_srad": 0.121, "koi_ror": 0.086,
        "ra": 346.6, "dec": -5.04, "koi_score": 0.9,
    }),
    ("Kepler-186f (1.11 Re)", {
        "koi_period": 129.9, "koi_depth": 282, "koi_duration": 6.2,
        "koi_steff": 3788, "koi_srad": 0.472, "koi_ror": 0.021,
        "ra": 282.95, "dec": 51.76, "koi_score": 0.85,
    }),
    ("Kepler-22b demo (2.4 Re)", {
        "koi_period": 289.9, "koi_depth": 492, "koi_duration": 7.4,
        "koi_steff": 5793, "koi_srad": 0.98, "koi_ror": 0.048,
        "ra": 291.0, "dec": 48.14, "koi_score": 0.9,
    }),
    ("Small planet (no optional fields)", {
        "koi_period": 6.1, "koi_depth": 700, "koi_duration": 0.85,
        "koi_steff": 2566, "koi_srad": 0.121, "koi_ror": 0.086,
    }),
    ("Obvious false positive (deep eclipse, large SNR mismatch)", {
        "koi_period": 0.8, "koi_depth": 500000, "koi_duration": 2.0,
        "koi_steff": 5500, "koi_srad": 1.0, "koi_ror": 0.45,
        "ra": 290.0, "dec": 45.0, "koi_score": 0.01,
    }),
]

print("=" * 80)
print("PREDICTION TEST RESULTS")
print("=" * 80)
for name, data in tests:
    r = requests.post(URL, json=data)
    d = r.json()
    if "error" in d:
        print(f"  {name}: ERROR - {d['error']}")
        continue
    pred = d["prediction"]
    prob = d["confirmation_probability"]
    radius = d["predicted_radius"]
    ptype = d["planet_type"]
    source = d["radius_source"]
    print(f"  {name}")
    print(f"    -> {pred} ({prob:.1f}%)  R={radius} Re [{ptype}] (radius: {source})")
    print()
