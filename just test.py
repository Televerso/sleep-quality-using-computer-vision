from collections import Counter

a = dict()

a[1] = "WAKE"

a[50] = "NREM"
a[150] = "REM"
a[180] = "NREM"

a[210] = "REM"
a[300] = "NREM"
a[600] = "REM"
a[660] = "NREM"
a[900] = "WAKE"
a[990] = "NREM"
a[1200] = "WAKE"

print(a)
vals = a.values()

print(Counter(vals)["WAKE"])