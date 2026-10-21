/**
 * FloodScope AI — Web Dashboard Client Application
 * Member 3: Application & Integration
 *
 * Handles:
 * 1. Preset sample vs file upload mode switching.
 * 2. Threshold slider, algorithm select, and morphology controls.
 * 3. Calling FastAPI backend (/api/detect/sample and /api/detect/upload).
 * 4. Rendering multi-view visualizations (overlay, mask, diff, side-by-side, diagnostics).
 * 5. Displaying quantitative metrics strictly computed over valid pixels.
 * 6. Managing loading and error states gracefully.
 */

document.addEventListener("DOMContentLoaded", () => {
  // --- DOM Elements ---
  const tabSampleBtn = document.getElementById("tab-sample-btn");
  const tabUploadBtn = document.getElementById("tab-upload-btn");
  const modeSampleContent = document.getElementById("mode-sample-content");
  const modeUploadContent = document.getElementById("mode-upload-content");

  const thresholdSlider = document.getElementById("threshold-slider");
  const thresholdVal = document.getElementById("threshold-val");
  const presetChips = document.querySelectorAll(".btn-chip");
  const methodSelect = document.getElementById("method-select");
  const morphToggle = document.getElementById("morph-toggle");
  const kernelSelect = document.getElementById("kernel-select");
  const kernelGroup = document.getElementById("kernel-group");

  const runBtn = document.getElementById("run-btn");
  const runBtnText = document.getElementById("run-btn-text");
  const runSpinner = document.getElementById("run-spinner");

  const uploadBefore = document.getElementById("upload-before");
  const uploadAfter = document.getElementById("upload-after");
  const uploadMask = document.getElementById("upload-mask");
  const beforeFileName = document.getElementById("before-file-name");
  const afterFileName = document.getElementById("after-file-name");
  const maskFileName = document.getElementById("mask-file-name");

  // Metrics DOM
  const mTotalPixels = document.getElementById("m-total-pixels");
  const mValidPixels = document.getElementById("m-valid-pixels");
  const mChangedPixels = document.getElementById("m-changed-pixels");
  const mChangePct = document.getElementById("m-change-pct");
  const metricStatus = document.getElementById("metric-status");

  // Visualizer Tab Buttons
  const viewTabBtns = document.querySelectorAll(".view-tab-btn");
  const viewPanes = document.querySelectorAll(".view-pane");

  // Image Elements & Placeholders
  const imgOverlay = document.getElementById("img-overlay");
  const imgMask = document.getElementById("img-mask");
  const imgDiff = document.getElementById("img-diff");
  const imgPre = document.getElementById("img-pre");
  const imgPost = document.getElementById("img-post");
  const imgCheckerboard = document.getElementById("img-checkerboard");
  const imgDiagBlend = document.getElementById("img-diag-blend");
  const imgOverlapMask = document.getElementById("img-overlap-mask");
  const imgDiagSbs = document.getElementById("img-diag-sbs");

  const placeholderOverlay = document.getElementById("placeholder-overlay");
  const placeholderMask = document.getElementById("placeholder-mask");
  const placeholderDiff = document.getElementById("placeholder-diff");
  const placeholderPre = document.getElementById("placeholder-pre");
  const placeholderPost = document.getElementById("placeholder-post");
  const placeholderChk = document.getElementById("placeholder-chk");
  const placeholderBlend = document.getElementById("placeholder-blend");
  const placeholderOverlap = document.getElementById("placeholder-overlap");
  const placeholderSbs = document.getElementById("placeholder-sbs");

  // Metadata Footer & Alerts
  const metaDims = document.getElementById("meta-dims");
  const metaAlgo = document.getElementById("meta-algo");
  const metaRunId = document.getElementById("meta-run-id");
  const errorAlert = document.getElementById("error-alert");
  const errorMsg = document.getElementById("error-msg");
  const alertCloseBtn = document.getElementById("alert-close-btn");

  let currentMode = "sample"; // 'sample' or 'upload'

  // --- Mode Switching ---
  tabSampleBtn.addEventListener("click", () => {
    currentMode = "sample";
    tabSampleBtn.classList.add("active");
    tabUploadBtn.classList.remove("active");
    modeSampleContent.classList.add("active");
    modeUploadContent.classList.remove("active");
  });

  tabUploadBtn.addEventListener("click", () => {
    currentMode = "upload";
    tabUploadBtn.classList.add("active");
    tabSampleBtn.classList.remove("active");
    modeUploadContent.classList.add("active");
    modeSampleContent.classList.remove("active");
  });

  // --- Parameter Controls ---
  thresholdSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value).toFixed(1);
    thresholdVal.textContent = val;
    presetChips.forEach((chip) => {
      chip.classList.toggle("active", parseFloat(chip.dataset.val) === parseFloat(val));
    });
  });

  presetChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const val = parseFloat(chip.dataset.val);
      thresholdSlider.value = val;
      thresholdVal.textContent = val.toFixed(1);
      presetChips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
    });
  });

  morphToggle.addEventListener("change", (e) => {
    kernelGroup.style.display = e.target.checked ? "block" : "none";
  });

  // --- File Input Labels ---
  uploadBefore.addEventListener("change", (e) => {
    const file = e.target.files[0];
    beforeFileName.textContent = file ? `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)` : "No file selected (Max 25MB)";
  });

  uploadAfter.addEventListener("change", (e) => {
    const file = e.target.files[0];
    afterFileName.textContent = file ? `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)` : "No file selected (Max 25MB)";
  });

  uploadMask.addEventListener("change", (e) => {
    const file = e.target.files[0];
    maskFileName.textContent = file ? `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)` : "Optional";
  });

  // --- View Tabs ---
  viewTabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetView = btn.dataset.view;
      viewTabBtns.forEach((b) => b.classList.remove("active"));
      viewPanes.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(`pane-${targetView}`);
      if (targetPane) targetPane.classList.add("active");
    });
  });

  alertCloseBtn.addEventListener("click", () => {
    errorAlert.classList.add("hidden");
  });

  function showError(msg) {
    errorMsg.textContent = msg;
    errorAlert.classList.remove("hidden");
  }

  function clearError() {
    errorAlert.classList.add("hidden");
  }

  function setLoading(isLoading) {
    runBtn.disabled = isLoading;
    runBtnText.textContent = isLoading ? "Processing Imagery..." : "Execute Candidate Detection";
    runSpinner.classList.toggle("hidden", !isLoading);
  }

  function renderImage(imgElem, placeholderElem, url) {
    if (!url) {
      imgElem.src = "";
      imgElem.style.display = "none";
      if (placeholderElem) placeholderElem.style.display = "flex";
      return;
    }
    // Add cache buster for run-specific freshness
    const bustUrl = url.includes("?") ? `${url}&_t=${Date.now()}` : `${url}?_t=${Date.now()}`;
    imgElem.onload = () => {
      imgElem.style.display = "block";
      if (placeholderElem) placeholderElem.style.display = "none";
    };
    imgElem.onerror = () => {
      imgElem.style.display = "none";
      if (placeholderElem) {
        placeholderElem.style.display = "flex";
        placeholderElem.innerHTML = `<span>Failed to load artifact: ${url}</span>`;
      }
    };
    imgElem.src = bustUrl;
  }

  function updateMetrics(summary) {
    if (!summary) return;
    mTotalPixels.textContent = (summary.total_pixels || 0).toLocaleString();
    mValidPixels.textContent = (summary.valid_pixels || 0).toLocaleString();
    mChangedPixels.textContent = (summary.changed_pixels || 0).toLocaleString();
    mChangePct.textContent = `${(summary.valid_area_change_pct || 0).toFixed(2)}%`;

    metricStatus.textContent = "● Processed & Verified";
    metricStatus.className = "status-indicator success";
  }

  // --- Execution Handler ---
  async function runDetection() {
    clearError();
    setLoading(true);

    const threshold = parseFloat(thresholdSlider.value);
    const method = methodSelect.value;
    const applyMorph = morphToggle.checked;
    const morphKernel = parseInt(kernelSelect.value, 10);

    try {
      let response;
      if (currentMode === "sample") {
        // Execute Sample Detection
        const payload = {
          sample_id: "SpaceNet8_Germany_0_16_67",
          threshold: threshold,
          method: method,
          apply_morphology: applyMorph,
          morph_kernel_size: morphKernel,
        };

        response = await fetch("/api/detect/sample", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      } else {
        // Execute Upload Detection
        if (!uploadBefore.files[0] || !uploadAfter.files[0]) {
          showError("Please select both Before and After images for upload.");
          setLoading(false);
          return;
        }

        const formData = new FormData();
        formData.append("before_image", uploadBefore.files[0]);
        formData.append("after_image", uploadAfter.files[0]);
        if (uploadMask.files[0]) {
          formData.append("overlap_mask", uploadMask.files[0]);
        }
        formData.append("threshold", threshold);
        formData.append("method", method);
        formData.append("apply_morphology", applyMorph);
        formData.append("morph_kernel_size", morphKernel);

        response = await fetch("/api/detect/upload", {
          method: "POST",
          body: formData,
        });
      }

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || data.error || `HTTP ${response.status} error occurred.`);
      }

      // Populate Visualizations
      const arts = data.artifacts || {};
      renderImage(imgOverlay, placeholderOverlay, arts.overlay_url);
      renderImage(imgMask, placeholderMask, arts.change_mask_url);
      renderImage(imgDiff, placeholderDiff, arts.difference_image_url);
      renderImage(imgPre, placeholderPre, arts.pre_image_url);
      renderImage(imgPost, placeholderPost, arts.post_image_url);

      // Diagnostics (if sample baseline or provided)
      if (currentMode === "sample") {
        renderImage(imgCheckerboard, placeholderChk, "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_alignment_checkerboard.png");
        renderImage(imgDiagBlend, placeholderBlend, "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_alignment_overlay.png");
        renderImage(imgOverlapMask, placeholderOverlap, "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_overlap_mask.png");
        renderImage(imgDiagSbs, placeholderSbs, "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_alignment_side_by_side.png");
      }

      // Update Summary & Metadata
      updateMetrics(data.summary);
      if (data.metadata && data.metadata.dimensions) {
        metaDims.textContent = `${data.metadata.dimensions[0]} × ${data.metadata.dimensions[1]} px`;
      }
      metaAlgo.textContent = method === "grayscale_diff" ? "Grayscale Luminance Difference" : "RGB Euclidean Distance";
      metaRunId.textContent = data.run_id || "default";

    } catch (err) {
      showError(err.message || "Failed to execute change detection.");
    } finally {
      setLoading(false);
    }
  }

  runBtn.addEventListener("click", runDetection);

  // --- Initial Page Load ---
  async function initializeApp() {
    try {
      const healthRes = await fetch("/api/health");
      if (healthRes.ok) {
        const health = await healthRes.json();
        document.getElementById("opencv-badge").textContent = `OpenCV ${health.opencv_version}`;
        document.getElementById("python-badge").textContent = `Python ${health.python_version}`;
      }

      // Auto-load default baseline for immediate visual presentation
      await runDetection();
    } catch (e) {
      console.warn("Auto-initialization notice:", e);
    }
  }

  initializeApp();
});
