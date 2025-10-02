import math, awkward as ak

def summarize_ak(arr, name="arr"):
    print(f"\n=== {name} ===")
    print("type:", ak.type(arr))
    N = len(arr)
    print("entries:", N)
    for f in ak.fields(arr):
        col = arr[f]
        t = ak.type(col)
        print(f"\n[{f}] type: {t}")
        # If it's a list -> multiplicities per entry
        try:
            lens = ak.num(col, axis=-1)
            if ak.type(lens).to_list().endswith("int64") or ak.type(lens).to_list().endswith("int32"):
                m = (ak.min(lens, mask_identity=True), ak.mean(lens), ak.max(lens, mask_identity=True))
                print(f"  lengths (min/mean/max): {m}")
        except Exception:
            pass
        # If it’s a record of numerics (e.g., positions), recurse a level
        if isinstance(t, ak.types.RecordType):
            for sub in ak.fields(col):
                v = col[sub]
                if ak.is_none(v, axis=None).tolist() is True:
                    continue
                try:
                    flat = ak.flatten(v, axis=None)
                    rng = (ak.min(flat, mask_identity=True), ak.mean(flat, axis=None), ak.max(flat, mask_identity=True))
                    print(f"  .{sub} range (min/mean/max): {rng}")
                except Exception:
                    pass
        # Numeric leaf(s): flatten and show range
        try:
            flat = ak.flatten(col, axis=None)
            if ak.is_numeric(flat, axis=None):
                rng = (ak.min(flat, mask_identity=True), ak.mean(flat, axis=None), ak.max(flat, mask_identity=True))
                print(f"  values (min/mean/max): {rng}")
        except Exception:
            pass