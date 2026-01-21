"""
Created on Wed Nov 12 16:51:02 2025

Single-compartment BEM

Volumetric source space restricted to occipital sensors (306 --> 72) within 7cm distance

Inverse solution by MNE

Masking source estimates to cerebellar segmentation

Comparison of source estimate activity by RMS and PVE cerebellar vs. occipital-cortical sources

@author: tarkiav1
"""

import mne, os, conpy
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import nibabel as nib
from mne.minimum_norm import make_inverse_resolution_matrix, get_point_spread, get_cross_talk
from mne.bem import make_watershed_bem
from mne.channels.layout import Layout
from mne.channels import read_layout
from mne.rank import compute_rank

# %%

root = "/m/nbe/scratch/artefact_sync"
for path, dirs, files in os.walk(root):
    for f in files:
        if f.endswith("-src.fif"):
            print(os.path.join(path, f))

# %%

out_fig_dir = "Thesis/Figures/STC"
os.makedirs(out_fig_dir, exist_ok=True)

folder_out = "/m/nbe/scratch/artefact_sync/tarkiav1/"

subject = "pilot03"
subjects_dir = "/m/nbe/scratch/artefact_sync/mri/"  # parent directory, NOT the /mri subfolder
os.environ["SUBJECTS_DIR"] = subjects_dir

subject_triux = subject

if subject == "pilot01":
    date = 240216

elif subject == "pilot04":
    date = 250127

else:
    date = 240625

# Load raw data
raw_fname = f'{folder_out}preprocessed_signals/{subject_triux}_raw.fif'  # preprocessed
raw = mne.io.read_raw_fif(raw_fname, preload=False)
info = raw.info

n_jobs = 8

# Source modeling parameters
conductivity = (0.3,)
source_space_type = 'volumetric'
pos = 5.0
lambda2 = 0.11111
method = 'MNE'  # mne
spacing = 2
cerebellum_subsampling = 'dense'
visualize = True
dist = 0.05  # 5cm? The minimum distance between a vertex and the nearest sensor (in meters). All vertices for which the distance to the nearest sensor exceeds this limit are discarded.

depth = 0.8
# %% Pick sensors

layout = read_layout("Vectorview-all")

left_occ = mne.read_vectorview_selection('Left-occipital', info=info)
right_occ = mne.read_vectorview_selection('Right-occipital', info=info)
picks_post = left_occ + right_occ

picks_clean = [ch.replace(' ', '') for ch in picks_post]
picks_final = [ch for ch in picks_clean if ch in raw.ch_names]  # intersect

posterior_raw = raw.copy().pick(picks_final)

# indices
posterior_idx = mne.pick_channels(raw.ch_names, picks_final)
posterior_info = mne.pick_info(info, sel=posterior_idx)

### ONLY MAG SNESORS?

picks_mag = mne.pick_types(info, meg='mag')  # indices of all mags
mag_names = [raw.ch_names[i] for i in picks_mag]

posterior_mag_names = [ch for ch in picks_clean if ch in mag_names]

posterior_raw_mag = raw.copy().pick(posterior_mag_names)
posterior_idx_mag = mne.pick_channels(raw.ch_names, posterior_mag_names)
posterior_info_mag = mne.pick_info(info, sel=posterior_idx_mag)

# %% Coregister
# from mne.coreg import Coregistration

try:
    if subject == "pilot04":
        trans_fname = "/m/nbe/scratch/artefact_sync/mri/pilot04/bem/pilot04-trans.fif"
        trans = mne.read_trans(trans_fname)
    else:
        # these pilots have .trans files already
        trans_fname = f"/m/nbe/scratch/artefact_sync/triux/processed/{subject}/{date}/trans/{subject}-trans.fif"
        trans = mne.read_trans(trans_fname)
except FileNotFoundError:
    print("File not found. Coregistering...")
    mne.gui.coregistration(subject=subject, inst=raw_fname, subjects_dir=subjects_dir, block=False)

# create new conda environment ml scicomp-python-env
# from mne
# coregister

# or load earlier version

# %% coreg check

from collections import Counter

FIFF = mne.io.constants.FIFF

dig = raw.info["dig"] or []
print("n_dig:", len(dig))
print("dig kinds:", Counter(d["kind"] for d in dig))

print("n_cardinal (fiducials):", sum(d["kind"] == FIFF.FIFFV_POINT_CARDINAL for d in dig))
print("n_extra (headshape):", sum(d["kind"] == FIFF.FIFFV_POINT_EXTRA for d in dig))
print("n_hpi:", sum(d["kind"] == FIFF.FIFFV_POINT_HPI for d in dig))

fid = {d["ident"]: d["r"] for d in dig if d["kind"] == FIFF.FIFFV_POINT_CARDINAL}

lpa = fid[FIFF.FIFFV_POINT_LPA]
nas = fid[FIFF.FIFFV_POINT_NASION]
rpa = fid[FIFF.FIFFV_POINT_RPA]

print("LPA–RPA (m):", np.linalg.norm(lpa - rpa))
print("NAS–LPA (m):", np.linalg.norm(nas - lpa))
print("NAS–RPA (m):", np.linalg.norm(nas - rpa))

# %% BEM

# os.makedirs("Thesis/pilot04", exist_ok=True)

# Watershed BEM
# try:
#    make_watershed_bem(subject, subjects_dir=subjects_dir)
# except RuntimeError:
#    pass

# Visualize the BEM surfaces
mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir, brain_surfaces='white', orientation='coronal')

# BEM model
print("Making BEM model...")
bem_model_fname = f"{folder_out}Source_est/BEM/{subject}-bem_model.fif"
try:
    model = mne.read_bem_surfaces(bem_model_fname)
except FileNotFoundError:
    model = mne.make_bem_model(subject, conductivity=(0.3,), subjects_dir=subjects_dir, ico=5)
    mne.write_bem_surfaces(bem_model_fname, model, overwrite=True)

print("Making BEM solution...")

try:
    if subject == "pilot04":
        bem_sol_fname = f"{folder_out}Source_est/BEM/{subject}-bem-sol.fif"
        bem_sol = mne.read_bem_solution(bem_sol_fname)
    else:
        bem_sol_fname = f"/m/nbe/scratch/artefact_sync/triux/processed/{subject}/{date}/forward/{subject}-bem-sol.fif"
        bem_sol = mne.read_bem_solution(bem_sol_fname)
    print(f"Loaded existing BEM solution from {bem_sol_fname}")

except FileNotFoundError:
    bem_sol = mne.make_bem_solution(model)
    mne.write_bem_solution(bem_sol_fname, bem_sol, overwrite=True)

# %% Source space
# Volumetric source space

# os.makedirs("Source_est/Source_space", exist_ok=True)
print("Setting up source space...")

try:
    if subject == "pilot04":
        src_fname = f"{folder_out}Source_est/Source_space/{subject}_{source_space_type}-src.fif"
        src = mne.read_source_spaces(src_fname)
    else:
        src_fname = f"/m/nbe/scratch/artefact_sync/triux/processed/{subject}/{date}/forward/{subject}_{source_space_type}-src.fif"
        src = mne.read_source_spaces(src_fname)
    print(f"Loaded existing source space from {src_fname}")

except FileNotFoundError:
    print(f"No existing source space found, creating a new {source_space_type} source space...")

    if source_space_type == 'volumetric':
        src = mne.setup_volume_source_space(subject=subject, subjects_dir=subjects_dir, pos=pos,
                                            bem=bem_sol, n_jobs=n_jobs)

    mne.write_source_spaces(f"{folder_out}Source_est/Source_space/{subject}{source_space_type}-src.fif", src,
                            overwrite=True)
    print(f"Saved source space to {src_fname}")

# %% Forward solution

# os.makedirs("Source_est/Forward", exist_ok=True)
print("Forward solution...")

try:
    if subject == "pilot04":
        forward_fname = f"{folder_out}Source_est/Forward/{subject}-fwd.fif"
        fwd = mne.read_forward_solution(forward_fname)
    else:
        forward_fname = f"/m/nbe/scratch/artefact_sync/triux/processed/{subject}/{date}/forward/{subject}_{source_space_type}-fwd.fif"
        fwd = mne.read_forward_solution(forward_fname)
    print(f"Loaded existing forward solution from {forward_fname}")

except FileNotFoundError:
    fwd = mne.make_forward_solution(info, trans, src, bem_sol, mindist=3.0, n_jobs=1)
    mne.write_forward_solution(forward_fname, fwd, overwrite=True)

# %% Restrict source space

print("Sensor space restriction...")

try:
    if subject == "pilot04":
        forward_r_fname = f"{folder_out}/Source_est/Forward/{subject}_{source_space_type}_restrict-fwd.fif"
        fwd_restrict = mne.read_forward_solution(forward_r_fname)
    else:
        forward_r_fname = f"/m/nbe/scratch/artefact_sync/triux/processed/{subject}/{date}/forward/{subject}_{source_space_type}_restrict-fwd.fif"
        fwd_restrict = mne.read_forward_solution(forward_r_fname)
except FileNotFoundError:
    fwd_restrict = conpy.restrict_forward_to_sensor_range(fwd, dist, picks=posterior_idx_mag, verbose=None)
    mne.write_forward_solution(forward_r_fname, fwd_restrict, overwrite=True)
print(f"Loaded restricted forward model from {forward_r_fname}.")

src_full = fwd["src"]
src_rest = fwd_restrict["src"]

print("n vertices (full):      ", sum(s["nuse"] for s in src_full))
print("n vertices (restricted):", sum(s["nuse"] for s in src_rest))

# interactive 3D source space plots
# src_full.plot(subject=subject, subjects_dir=subjects_dir, title="Source space (full)")
# src_rest.plot(subject=subject, subjects_dir=subjects_dir, title="Source space (restricted)")

# %% Noise covariance from epochs

epochs_orig = mne.read_epochs(f"{folder_out}Evoked_responses/epochs_and_evokeds/{subject_triux}_eog-epo.fif")
epochs_ica = mne.read_epochs(f"{folder_out}Evoked_responses/epochs_and_evokeds/{subject_triux}_eog_ICA-epo.fif")

epochs_orig_for_cov = epochs_orig.copy().apply_baseline(baseline=(-1.5, -0.05))
epochs_ica_for_cov = epochs_ica.copy().apply_baseline(baseline=(-1.5, -0.05))

evoked_orig = \
mne.read_evokeds(f"{folder_out}Evoked_responses/epochs_and_evokeds/{subject_triux}_evoked_eog_combined-ave.fif")[0]
evoked_ica = \
mne.read_evokeds(f"{folder_out}Evoked_responses/epochs_and_evokeds/{subject_triux}_evoked_eog_combined_ICA-ave.fif")[0]

methods = ['shrunk', 'diagonal_fixed', 'empirical']

cov_orig = mne.compute_covariance(epochs_orig_for_cov, method=methods, tmin=-1.5, tmax=-0.05, rank='info')
cov_ica = mne.compute_covariance(epochs_ica_for_cov, method=methods, tmin=-1.5, tmax=-0.05, rank='info')

# compute rank from covariance matrix at max lower
# info rank is theoretical max (info) --> small enough
# roughly 70

rank_orig = compute_rank(cov_orig, rank='info', info=epochs_orig.info)
rank_ica = compute_rank(cov_ica, rank='info', info=epochs_ica.info)

print(f"Rank Orig: {rank_orig}")
print(f"Rank ICA:  {rank_ica}")

figs_orig = cov_orig.plot(epochs_orig.info, proj=True, exclude='bads')
figs_ica = cov_ica.plot(epochs_ica.info, proj=True, exclude='bads')

# Save Original Covariance Figures
for i, fig in enumerate(figs_orig):
    # If there are multiple figures (e.g., Mag and Grad),
    # we add an index to the filename
    suffix = f"_{i}" if len(figs_orig) > 1 else ""
    fig.savefig(f"Thesis/Figures/STC/{subject}_cov_orig{suffix}.pdf",
                dpi=300, bbox_inches="tight")

# Save ICA Covariance Figures
for i, fig in enumerate(figs_ica):
    suffix = f"_{i}" if len(figs_ica) > 1 else ""
    fig.savefig(f"Thesis/Figures/STC/{subject}_cov_ica{suffix}.pdf",
                dpi=300, bbox_inches="tight")

# %% Inverse solution (for both full and restricted src) & Visualization

response_peaks = {
    "pilot01": 0.12,
    "pilot02": 0.06,
    "pilot03": 0.0,
    "pilot04": 0.15}  # need to change

visualize = 'activation'  # 'activation'

if source_space_type == 'surface':
    fwd_restrict = mne.convert_forward_solution(fwd_restrict, surf_ori=True, force_fixed=True, copy=True)
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, copy=True)
    fixed = True
else:
    fixed = False

inv_restrict_orig_fname = f"{folder_out}Source_est/Inverse/restricted_orig_{subject}-inv.fif"
inv_restrict_ica_fname = f"{folder_out}Source_est/Inverse/restricted_ica_{subject}-inv.fif"
inv_full_orig_fname = f"{folder_out}Source_est/Inverse/{subject}-inv.fif"

force_recompute = True

if force_recompute:

    inv_restrict_orig = mne.minimum_norm.make_inverse_operator(info=evoked_orig.info,
                                                               forward=fwd_restrict,
                                                               noise_cov=cov_orig,
                                                               rank=rank_orig,
                                                               depth=depth,
                                                               fixed=fixed,
                                                               loose='auto' if not fixed else None
                                                               )

    inv_restrict_ica = mne.minimum_norm.make_inverse_operator(info=evoked_ica.info,
                                                              forward=fwd_restrict,
                                                              noise_cov=cov_ica,
                                                              rank=rank_ica,
                                                              depth=depth,
                                                              fixed=fixed,
                                                              loose='auto' if not fixed else None
                                                              )

    inv_full_orig = mne.minimum_norm.make_inverse_operator(info=evoked_orig.info,
                                                           forward=fwd,
                                                           noise_cov=cov_orig,
                                                           rank=rank_orig,
                                                           depth=depth,
                                                           fixed=fixed,
                                                           loose='auto' if not fixed else None
                                                           )

    mne.minimum_norm.write_inverse_operator(inv_restrict_orig_fname, inv_restrict_orig, overwrite=True)
    mne.minimum_norm.write_inverse_operator(inv_restrict_ica_fname, inv_restrict_ica, overwrite=True)
    mne.minimum_norm.write_inverse_operator(inv_full_orig_fname, inv_full_orig, overwrite=True)
    print("Saved inverse operators")

else:
    try:
        inv_restrict_orig = mne.minimum_norm.read_inverse_operator(inv_restrict_orig_fname)
        inv_restrict_ica = mne.minimum_norm.read_inverse_operator(inv_restrict_ica_fname)
        inv_full_orig = mne.minimum_norm.read_inverse_operator(inv_full_orig_fname)
        # inv = mne.minimum_norm.read_inverse_operator(inv_fname)

    except FileNotFoundError:
        inv_restrict_orig = mne.minimum_norm.make_inverse_operator(info=evoked_orig.info,
                                                                   forward=fwd_restrict,
                                                                   noise_cov=cov_orig,
                                                                   rank=rank_orig,
                                                                   depth=depth,
                                                                   fixed=fixed,
                                                                   loose='auto' if not fixed else None
                                                                   )

        inv_restrict_ica = mne.minimum_norm.make_inverse_operator(info=evoked_ica.info,
                                                                  forward=fwd_restrict,
                                                                  noise_cov=cov_ica,
                                                                  rank=rank_ica,
                                                                  depth=depth,
                                                                  fixed=fixed,
                                                                  loose='auto' if not fixed else None
                                                                  )

        inv_full_orig = mne.minimum_norm.make_inverse_operator(info=evoked_orig.info,
                                                               forward=fwd,
                                                               noise_cov=cov_orig,
                                                               rank=rank_orig,
                                                               depth=depth,
                                                               fixed=fixed,
                                                               loose='auto' if not fixed else None
                                                               )

        # Save inverse operator
        mne.minimum_norm.write_inverse_operator(inv_restrict_orig_fname, inv_restrict_orig, overwrite=True)
        mne.minimum_norm.write_inverse_operator(inv_restrict_ica_fname, inv_restrict_ica, overwrite=True)
        mne.minimum_norm.write_inverse_operator(inv_full_orig_fname, inv_full_orig, overwrite=True)
        print("Saved inverse operators")

# Apply inverse to restricted forward solution
stc_orig_restrict = mne.minimum_norm.apply_inverse(
    evoked_orig,
    inv_restrict_orig,
    lambda2=lambda2,
    method=method,
)

stc_ica_restrict = mne.minimum_norm.apply_inverse(
    evoked_ica,
    inv_restrict_ica,
    lambda2=lambda2,
    method=method,
)

# Apply inverse to full forward solution
stc_orig_full = mne.minimum_norm.apply_inverse(
    evoked_orig,
    inv_full_orig,
    lambda2=lambda2,
    method=method,
)

"""stc_ica_full = mne.minimum_norm.apply_inverse(
        evoked_ica,
        inv_full,
        lambda2=lambda2,
        method=method,
        )"""

stc_orig_restrict_fname = f"{folder_out}Source_est/STC/{subject}_original_restricted"
stc_orig_restrict.save(stc_orig_restrict_fname, overwrite=True)

stc_ica_restrict_fname = f"{folder_out}Source_est/STC/{subject}_ICA_restricted"
stc_ica_restrict.save(stc_ica_restrict_fname, overwrite=True)

stc_orig_full_fname = f"{folder_out}Source_est/STC/{subject}_original_full"
stc_orig_full.save(stc_orig_restrict_fname, overwrite=True)

"""stc_ica_restrict_fname = f"{folder_out}Source_est/STC/{subject}_ICA_full"
stc_ica_restrict.save(stc_ica_restrict_fname, overwrite=True)"""

print("Saved STCs")

# %% Visualize source estimates

clim = dict(kind="value", lims=[0, 3e-11, 7e-11])

src = inv_full_orig["src"][0]
rr = src["rr"]  # shape (n_vertices, 3), in meters
inuse = src["inuse"]  # 1 = valid source, 0 = excluded

rr_use = rr[inuse.astype(bool)]
use_idx = np.where(inuse)[0]

y_coords = rr_use[:, 1]  # anterior–posterior axis

# Choose, e.g., the most posterior 10%
threshold = np.percentile(y_coords, 10)
posterior_mask = y_coords <= threshold

posterior_vertices = use_idx[posterior_mask]

y = rr_use[:, 1]
z = rr_use[:, 2]

posterior_idx = posterior_vertices[np.argmin(rr[posterior_vertices, 1])]
vertex_index = posterior_idx

vertex_coords = rr[vertex_index]
print(f"Posterior vertex {vertex_index} at {vertex_coords} (m)")

# Full source space original and ICA
brain_orig_full = stc_orig_full.plot(
    src=inv_full_orig["src"],
    subject=subject,
    subjects_dir=subjects_dir,
    mode='glass_brain',
    initial_time=response_peaks[subject],
    # initial_pos=vertex_coords,
    clim='auto',
)

"""brain_ica_full= stc_ica_full.plot(
        src=inv_full["src"],
        subject=subject,
        subjects_dir=subjects_dir,
        mode='stat_map',
        initial_time=response_peaks[subject],
        clim='auto',
    )"""

# Restricted source space original and ICA
brain_orig_restrict = stc_orig_restrict.plot(
    src=inv_restrict_orig["src"],
    subject=subject,
    subjects_dir=subjects_dir,
    mode='glass_brain',  # 'glass_brain', 'stat_map'
    initial_time=response_peaks[subject],
    # initial_pos=vertex_coords,
    clim='auto',
)

brain_ica_restrict = stc_ica_restrict.plot(
    src=inv_restrict_ica["src"],
    subject=subject,
    subjects_dir=subjects_dir,
    mode='glass_brain',
    initial_time=response_peaks[subject],
    # initial_pos=vertex_coords,
    clim='auto',
)

"""brain_difference_restrict = stc_difference.plot(
        src=inv_restrict_orig["src"],
        subject=subject,
        subjects_dir=subjects_dir,
        mode='glass_brain',
        initial_time=response_peaks[subject],
        clim='auto',
    )"""

brain_orig_restrict.savefig(f"{out_fig_dir}/{subject}_source_est_orig.pdf", dpi=300, bbox_inches="tight")
brain_ica_restrict.savefig(f"{out_fig_dir}/{subject}_source_est_ica.pdf", dpi=300, bbox_inches="tight")
brain_orig_full.savefig(f"{out_fig_dir}/{subject}_source_est_full_orig.pdf", dpi=300, bbox_inches="tight")

# %% Restriction check

def describe_fwd(tag, fwd):
    src = fwd['src']
    n_verts = [s['nuse'] for s in src]
    print(f"{tag}: nuse per hemi = {n_verts}, total = {sum(n_verts)}")


print("=== SENSOR CHECK ===")
print("len(posterior_idx):", len(posterior_idx_mag))
print("posterior names:", [fwd['info']['ch_names'][p] for p in posterior_idx_mag[:10]])

print("\n=== RESTRICTION CHECK ===")
describe_fwd("original", fwd)
describe_fwd("restricted", fwd_restrict)

print("fwd shapes:",
      fwd['sol']['data'].shape, "->",
      fwd_restrict['sol']['data'].shape)

# %% Mask cerebellum to source estimate

from nibabel.affines import apply_affine
from mne.transforms import apply_trans, invert_transform
from nibabel.processing import resample_from_to

# When works, same for ICA data

# 1, Load segmentation
segmentation_dir = f"/m/nbe/scratch/artefact_sync/mri/cerebellum_seg/{subject}.nii.gz"
segmentation = nib.load(segmentation_dir)

# 2. MRI path
t1_path = f"/m/nbe/scratch/artefact_sync/mri/{subject}/mri/T1.mgz"
t1_img = nib.load(t1_path)

seg_on_t1 = resample_from_to(segmentation, t1_img, order=0)

t1_data = t1_img.get_fdata().astype(np.float32)
seg_data = seg_on_t1.get_fdata().astype(np.float32)

print("T1 shape:", t1_data.shape, "Seg (resampled) shape:", seg_data.shape)
print("Seg nonzero voxels:", np.count_nonzero(seg_data))

# Get mask data
# seg_data = segmentation.get_fdata(dtype=np.float32) #numpy array, shape (256, 256, 256)
seg_affine = segmentation.affine  # affine transformation (voxel-to-world mapping), dimensions (np.float32(1.0), np.float32(1.0), np.float32(1.0))
header = segmentation.header  # metadata

# Segmentation dimensions
# nx, ny, nz = seg_data.shape #segmentation coordinates, shape (256, 256, 256)
nx, ny, nz = t1_data.shape
print("Non-zero voxels in segmentation:", np.count_nonzero(seg_data))  # 183326
print("Unique values:", np.unique(seg_data)[:10])

# 2. Get source space
# Source spaces used for inverse
vol_src = inv_restrict_orig["src"][0]
print("Source coord_frame:", vol_src["coord_frame"])  # should be 4 (HEAD)

# Indices of the vertices in STC
stc_vertices = stc_orig_restrict.vertices[0]  # shape=(4480,)
print(f"Source count: {len(stc_vertices)}")  # 4480

rr_head_m = vol_src['rr'][stc_vertices]

# Transform HEAD -> MRI using inverse operator transform ---
mri_head_t = inv_restrict_orig['mri_head_t']  # MRI --> HEAD
head_mri_t = invert_transform(mri_head_t)  # HEAD --> MRI

rr_mri_m = apply_trans(head_mri_t, rr_head_m)  # still in meters, but now MRI coords

# convert to millimeters (to match NIfTI conventions)
rr_mri_mm = rr_mri_m * 1000.0

# 3. Map coordinates to voxels

# inv_affine = np.linalg.inv(seg_affine) # world to voxel
inv_affine = np.linalg.inv(t1_img.affine)
vox_coords = apply_affine(inv_affine, rr_mri_mm)
vox_indices = np.round(vox_coords).astype(int)
i, j, k = vox_indices.T  # source space coordinate indices

# 4. Create mask

# source points within MRI coordinates
mask_inside = (
        (i >= 0) & (i < nx) &
        (j >= 0) & (j < ny) &
        (k >= 0) & (k < nz)
)  # shape=(4480, 1)

print("Sources inside T1 volume:", np.sum(mask_inside))

mask_is_cerebellum = np.zeros_like(mask_inside, dtype=bool)
mask_is_cerebellum[mask_inside] = (
        seg_data[i[mask_inside], j[mask_inside], k[mask_inside]] != 0
)

# final mask: inside volume AND segmentation non-zero
final_mask = mask_inside & mask_is_cerebellum
final_mask = final_mask.ravel()  # 1D mask

print(f"Final Cerebellum sources: {np.sum(final_mask)}")  #
print(f"Original sources: {len(stc_vertices)}")  # 4480

# 5. Apply mask to source estimate

stc_cereb_orig = stc_orig_restrict.copy()
stc_cereb_orig._data = stc_cereb_orig.data[final_mask, :]
stc_cereb_orig.vertices[0] = stc_vertices[final_mask]

# use the same mask for ICA
stc_cereb_ica = stc_ica_restrict.copy()
stc_cereb_ica._data = stc_cereb_ica.data[final_mask, :]
stc_cereb_ica.vertices[0] = stc_ica_restrict.vertices[0][final_mask]


# %% Visualize cerebellum segmentation

# WHY DOES IT LOOK LIKE THERE ARE NO SOURCES INSIDE CEREBELLUM? - bad slice selection?

def pick_slice_max(mask3d, axis):
    counts = np.sum(mask3d != 0, axis=tuple(a for a in range(3) if a != axis))
    return int(np.argmax(counts))


# pick informative slices based on segmentation density
i0 = pick_slice_max(seg_data, axis=0)
j0 = pick_slice_max(seg_data, axis=1)
k0 = pick_slice_max(seg_data, axis=2)

valid_i = i[final_mask]
valid_j = j[final_mask]
valid_k = k[final_mask]

if len(valid_i) == 0:
    print("Zero sources found in mask. Cannot plot.")
else:
    # 2. Pick slices that actually contain the most sources
    # We use bincount to find which slice index appears most often in the valid sources
    i0 = np.argmax(np.bincount(valid_i))
    j0 = np.argmax(np.bincount(valid_j))
    k0 = np.argmax(np.bincount(valid_k))

    print(f"Best source slices -> Sagittal: {i0}, Coronal: {j0}, Axial: {k0}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))


    def show(ax, view, idx):
        # We increase the 'thickness' of the slice view to +/- 2 voxels
        # because sources might be slightly off the exact center of the slice.
        thickness = 2

        if view == "sag":  # i fixed -> show (j,k)
            bg = t1_data[idx, :, :].T
            sg = seg_data[idx, :, :].T

            # Check for sources within 'thickness' of this slice
            near = mask_inside & (np.abs(i - idx) <= thickness)

            x_all, y_all = j[near], k[near]
            x_cb, y_cb = j[near & mask_is_cerebellum], k[near & mask_is_cerebellum]

            ax.set_title("Sagittal", fontsize=18, fontweight="bold")
            # ax.set_xlabel("j (Posterior -> Anterior)", fontsize=14)
            # ax.set_ylabel("k (Inferior -> Superior)", fontsize=14)
            ax.set_xticks([0, 50, 100, 150, 200, 250]);
            ax.set_yticks([0, 50, 100, 150, 200, 250])
            ax.set_xticklabels(ax.get_xticks(), fontsize=14);
            ax.set_yticklabels(ax.get_yticks(), fontsize=14)


        elif view == "cor":  # j fixed -> show (i,k)
            bg = t1_data[:, idx, :].T
            sg = seg_data[:, idx, :].T
            near = mask_inside & (np.abs(j - idx) <= thickness)
            x_all, y_all = i[near], k[near]
            x_cb, y_cb = i[near & mask_is_cerebellum], k[near & mask_is_cerebellum]
            ax.set_title("Coronal", fontsize=18, fontweight="bold")
            # ax.set_xlabel("i (Right -> Left)",fontsize=14)
            # ax.set_ylabel("k", fontsize=14)
            ax.set_xticks([0, 50, 100, 150, 200, 250]);
            ax.set_yticks([0, 50, 100, 150, 200, 250])
            ax.set_xticklabels(ax.get_xticks(), fontsize=14);
            ax.set_yticklabels(ax.get_yticks(), fontsize=14)


        else:  # axial k fixed -> show (i,j)
            bg = t1_data[:, :, idx].T
            sg = seg_data[:, :, idx].T
            near = mask_inside & (np.abs(k - idx) <= thickness)
            x_all, y_all = i[near], j[near]
            x_cb, y_cb = i[near & mask_is_cerebellum], j[near & mask_is_cerebellum]
            ax.set_title("Axial", fontsize=18, fontweight="bold")
            # ax.set_xlabel("i", fontsize=14); ax.set_ylabel("j", fontsize=14)
            ax.set_xticks([0, 50, 100, 150, 200, 250]);
            ax.set_yticks([0, 50, 100, 150, 200, 250])
            ax.set_xticklabels(ax.get_xticks(), fontsize=14);
            ax.set_yticklabels(ax.get_yticks(), fontsize=14)

        ax.imshow(bg, origin="lower", cmap="gray")
        # Ensure the mask overlay matches the bg orientation
        ax.imshow(np.ma.masked_where(sg == 0, sg), origin="lower", cmap="autumn", alpha=0.3)

        ax.contour((sg > 0).astype(int), levels=[0.5], linewidths=2, colors="white")

        # Plot all sources (blue dots)
        ax.scatter(x_all, y_all, s=20, marker=".", c="cyan", alpha=0.5, label="All sources")
        # Plot cerebellum sources (red Xs)
        ax.scatter(x_cb, y_cb, s=50, marker="x", c="red", linewidth=1, label="Cerebellar sources")
        # if view == "axi": ax.legend(loc="upper right", fontsize='large')

    show(axes[0], "sag", i0)
    show(axes[1], "cor", j0)
    show(axes[2], "axi", k0)

    plt.tight_layout()
    plt.show()

fig.savefig(f"{out_fig_dir}/{subject}_cerebellum_segmentation.pdf", dpi=300, bbox_inches="tight")

# %% RMS over voxels in cerebellum

def rms_over_voxels(stc, global_scalar=False):
    """
    If global_scalar=False: Returns RMS over sensors for each time point (array).
    If global_scalar=True:  Returns a single scalar RMS for the entire epoch.
    """
    data = stc.data
    if global_scalar:
        # Standard deviation/RMS of all data points combined
        return np.sqrt(np.mean(data ** 2))
    else:
        # RMS across sensors at each time point
        return np.sqrt(np.mean(data ** 2, axis=0))


def compute_rms(stc_orig, stc_ica):
    # 1. Get RMS arrays for plotting (Function of Time)
    rms_orig_arr = rms_over_voxels(stc_orig, global_scalar=False)
    rms_ica_arr = rms_over_voxels(stc_ica, global_scalar=False)

    # 2. Get Global Scalars for PVE reporting
    rms_orig_global = rms_over_voxels(stc_orig, global_scalar=True)
    rms_ica_global = rms_over_voxels(stc_ica, global_scalar=True)

    # Calculate Global PVE (Scalar)
    global_pve = (1 - (rms_ica_global ** 2 / rms_orig_global ** 2)) * 100

    return rms_orig_arr, rms_ica_arr, global_pve

# Cerebellum sources
rms_cereb_orig, rms_cereb_ica, global_pve_cereb = compute_rms(stc_cereb_orig, stc_cereb_ica)
rms_brain_orig, rms_brain_ica, global_pve_brain = compute_rms(stc_orig_restrict, stc_ica_restrict)

# %% RMS and PVE

times = stc_orig_restrict.times  # seconds

# Logaritmic scale
eps = 1e-20

rms_cereb_orig_plot = np.maximum(rms_cereb_orig, eps)
rms_cereb_ica_plot = np.maximum(rms_cereb_ica, eps)

# Cerebellum
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 5))

ax.plot(times, rms_cereb_orig_plot, label='RMS before cardiac ICA', color='tab:red', lw=1)
ax.plot(times, rms_cereb_ica_plot, label='RMS after cardiac ICA', color='tab:blue', lw=1)
ax.set_yscale('log')
ax.set_ylabel('RMS amplitude (log scale)', fontsize=14, fontweight="bold")
ax.set_xlabel('Time (s)', fontsize=14, fontweight="bold")
# ax.set_xticks([-2, -1.5, -1.0, -0.5, 0.0, 0.5]) #ax.set_yticks([10e-11])
# ax.set_xticklabels(ax.get_xticks(), fontsize=14) #ax.set_yticklabels(ax.get_yticks(), fontsize=14)
ax.tick_params(axis="both", which="major", labelsize=14)
ax.tick_params(axis="both", which="minor", labelsize=12)
ax.axvline(0, color='k', ls='--', lw=2, label='Saccade onset')
ax.set_title('Cerebellar RMS before and after cardiac ICA', fontsize=16, fontweight="bold")
ax.legend(frameon=False, loc='upper left', fontsize=14)
ax.grid(axis="both", alpha=0.3)
plt.tight_layout()

fig.savefig(f"{out_fig_dir}/{subject}_rms_reduction_cerebellum.pdf", dpi=300, bbox_inches='tight')
plt.show()

# Brain
rms_brain_orig_plot = np.maximum(rms_brain_orig, eps)
rms_brain_ica_plot = np.maximum(rms_brain_ica, eps)

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 5))

ax.plot(times, rms_brain_orig_plot, label='RMS before cardiac ICA', color='tab:red', lw=1)
ax.plot(times, rms_brain_ica_plot, label='RMS after cardiac ICA', color='tab:blue', lw=1)
ax.set_yscale('log')
ax.set_ylabel('RMS amplitude (log scale)', fontsize=14, fontweight="bold")
ax.set_xlabel('Time (s)', fontsize=14, fontweight="bold")
# ax.set_xticks([-2, -1.5, -1.0, -0.5, 0.0, 0.5]) #ax.set_yticks([4 * 10e-12, 6 * 10e-12, 10e-11, 2 * 10e-11])
# ax.set_xticklabels(ax.get_xticks(), fontsize=14) #ax.set_yticklabels(ax.get_yticks(), fontsize=14)
ax.tick_params(axis="both", which="major", labelsize=14)
ax.tick_params(axis="both", which="minor", labelsize=12)
ax.axvline(0, color='k', ls='--', lw=2, label='Saccade onset')
ax.set_title('Global RMS before and after cardiac ICA', fontsize=16, fontweight="bold")
ax.legend(frameon=False, loc='upper left', fontsize=14)
ax.grid(axis="both", alpha=0.3)
plt.tight_layout()

fig.savefig(f"{out_fig_dir}/{subject}_rms_reduction_brain.pdf", dpi=300, bbox_inches='tight')
plt.show()

# Compare PVEs
reduction_brain = 100 - global_pve_brain
reduction_cereb = 100 - global_pve_cereb

# Calculate the ratio of cleaning (Cerebellum vs Brain)
cleaning_ratio = global_pve_cereb / global_pve_brain

print(f"Global PVE cerebellum: {global_pve_cereb} and brain {global_pve_brain}")
# print(f"Reduction in Cerebellum: {reduction_cereb:.2f}%")
# print(f"Reduction in Brain: {reduction_brain:.2f}%")
print(f"Relative difference: {cleaning_ratio:2f}")