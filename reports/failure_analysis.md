# Failure Analysis

The final intent classifier was evaluated using 5-fold out-of-fold
predictions on the 200-example golden evaluation set.

Overall accuracy was 49.5%, resulting in 101 misclassified examples.

## Top Failure Modes

### 1. iCloud Backup vs Display/Input

Example:

> I want my fucking picture back pull it up on iCloud...

True intent: `icloud_backup`

Predicted intent: `display_input`

Hypothesis:

The classifier overweights generic device/photo vocabulary and does not
sufficiently distinguish photo recovery through iCloud from display or
input-related problems.

Potential improvement:

Add more iCloud recovery, synchronization, and backup examples to the
training set and explicitly separate cloud-storage/recovery language from
camera/display language.

---

### 2. Connectivity vs Apple Services

Example:

> you really need to fix this update or something because every time me
> and my friend try to FaceTime or call, the other person doesn’t get any
> notification...

True intent: `connectivity`

Predicted intent: `apple_services`

Hypothesis:

FaceTime and calling are strong Apple-service signals, while the actual
failure is related to communication/connectivity behavior.

Potential improvement:

Add boundary examples where the same Apple service appears in both
service-level and connectivity-level problems, and label according to the
primary failure mechanism.

---

### 3. Display/Input vs Battery

Example:

> The “done” button will not let me select my picture. It’s covered in the
> top right corner of the phone with the battery and wifi icons.

True intent: `display_input`

Predicted intent: `battery_charging`

Hypothesis:

The classifier is distracted by the explicit word "battery", despite the
actual issue being a blocked UI control.

Potential improvement:

Give more weight to the described failure action and UI interaction than
incidental device-status vocabulary.

---

### 4. iOS Update vs Connectivity

Example:

> Me eternally waiting for my iPhone to do ANYTHING after updating to
> #iOS11 #SLOWAF

True intent: `ios_update`

Predicted intent: `connectivity`

Hypothesis:

The model does not reliably distinguish an update-triggered device problem
from an actual network/connectivity problem.

Potential improvement:

Increase training examples where iOS versions are mentioned as context
rather than as the actual intent, while adding clearer update-failure
examples.

---

### 5. Apple Services vs Apps/App Store

Example:

> I went on iTunes and re downloaded it and it still isn’t in my library.

True intent: `apple_services`

Predicted intent: `apps_app_store`

Hypothesis:

The terms "downloaded" and "app" / application-like language overlap
strongly with the App Store intent, while iTunes is classified as an Apple
service.

Potential improvement:

Add explicit boundary examples for iTunes, Apple Music, iMessage,
FaceTime, and App Store workflows and consider a hierarchical classifier
for service-specific requests.

## General Observation

Most errors occur between semantically adjacent intents rather than between
completely unrelated categories.

The largest recurring boundaries are:

- `ios_update` ↔ `display_input`
- `connectivity` ↔ `display_input`
- `apple_services` ↔ `connectivity`
- `apple_services` ↔ `apps_app_store`

This suggests that the main limitation is not simply insufficient model
capacity. The taxonomy contains overlapping language and requires
understanding the primary failure mechanism.

## What We Would Improve Next

1. Expand the golden set, especially underrepresented intents.
2. Add deliberate boundary examples between frequently confused intents.
3. Use context more systematically when available.
4. Calibrate the classifier margin on a separate validation set.
5. Evaluate retrieval relevance independently from classifier predictions.