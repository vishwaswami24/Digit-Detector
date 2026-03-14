(function() {
    "use strict";

    const state = {
        isDrawing: false,
        lastX: 0,
        lastY: 0,
        hasInk: false,
        busy: false,
        correctionOpen: false,
        feedbackLocked: false,
        currentPrediction: null,
        currentImageData: "",
    };

    const elements = {
        canvas: document.getElementById("drawingCanvas"),
        clearBtn: document.getElementById("clearBtn"),
        predictBtn: document.getElementById("predictBtn"),
        rightBtn: document.getElementById("rightBtn"),
        wrongBtn: document.getElementById("wrongBtn"),
        retrainBtn: document.getElementById("retrainBtn"),
        digitDisplay: document.getElementById("digitDisplay"),
        confidenceValue: document.getElementById("confidenceValue"),
        confidenceFill: document.getElementById("confidenceFill"),
        topPredictions: document.getElementById("topPredictions"),
        probabilitiesGrid: document.getElementById("probabilitiesGrid"),
        predictionNarrative: document.getElementById("predictionNarrative"),
        feedbackPrompt: document.getElementById("feedbackPrompt"),
        correctionPanel: document.getElementById("correctionPanel"),
        digitSelector: document.getElementById("digitSelector"),
        feedbackSuccess: document.getElementById("feedbackSuccess"),
        totalFeedback: document.getElementById("totalFeedback"),
        correctionsCount: document.getElementById("correctionsCount"),
        agreementRate: document.getElementById("agreementRate"),
        averageConfidence: document.getElementById("averageConfidence"),
        healthBadge: document.getElementById("healthBadge"),
        healthDetail: document.getElementById("healthDetail"),
        retrainMeta: document.getElementById("retrainMeta"),
        loading: document.getElementById("loading"),
        loadingText: document.getElementById("loadingText"),
        errorMessage: document.getElementById("errorMessage"),
        canvasHint: document.getElementById("canvasHint"),
    };

    const ctx = elements.canvas.getContext("2d");

    function formatPercent(value) {
        return `${(value * 100).toFixed(1)}%`;
    }

    function showMessage(element, message) {
        element.textContent = message;
        element.hidden = false;
    }

    function hideMessage(element) {
        element.hidden = true;
        element.textContent = "";
    }

    function showError(message) {
        showMessage(elements.errorMessage, message);
    }

    function setBusy(isBusy, message) {
        state.busy = isBusy;
        elements.loading.hidden = !isBusy;
        if (message) {
            elements.loadingText.textContent = message;
        }
        refreshControlStates();
    }

    function refreshControlStates() {
        elements.clearBtn.disabled = state.busy;
        elements.predictBtn.disabled = state.busy || !state.hasInk;
        elements.retrainBtn.disabled = state.busy;

        const feedbackEnabled =
            Boolean(state.currentPrediction) && !state.feedbackLocked && !state.busy;
        elements.rightBtn.disabled = !feedbackEnabled;
        elements.wrongBtn.disabled = !feedbackEnabled;
    }

    function updateFeedbackPrompt(message) {
        elements.feedbackPrompt.textContent = message;
    }

    function closeCorrectionPanel() {
        state.correctionOpen = false;
        elements.correctionPanel.hidden = true;
        elements.wrongBtn.textContent = "Correct it";
    }

    function openCorrectionPanel() {
        state.correctionOpen = true;
        elements.correctionPanel.hidden = false;
        elements.wrongBtn.textContent = "Cancel";
    }

    function renderPrediction(prediction) {
        if (!prediction) {
            elements.digitDisplay.textContent = "--";
            elements.confidenceValue.textContent = "0%";
            elements.confidenceFill.style.width = "0%";
            elements.predictionNarrative.textContent =
                "Run a prediction to see how the model is reasoning.";
            elements.topPredictions.innerHTML =
                '<p class="muted-copy">Top alternatives appear after a prediction.</p>';
            elements.probabilitiesGrid.innerHTML =
                '<p class="muted-copy">Confidence bars update after each prediction.</p>';
            return;
        }

        elements.digitDisplay.textContent = prediction.digit;
        elements.confidenceValue.textContent = formatPercent(prediction.confidence);
        elements.confidenceFill.style.width = `${Math.max(prediction.confidence * 100, 4)}%`;

        if (prediction.confidence >= 0.85) {
            elements.predictionNarrative.textContent =
                "The model is strongly leaning toward this digit.";
        } else if (prediction.confidence >= 0.6) {
            elements.predictionNarrative.textContent =
                "The guess is reasonable, but the alternatives still matter.";
        } else {
            elements.predictionNarrative.textContent =
                "The model is unsure here, so a correction would be especially valuable.";
        }

        elements.topPredictions.innerHTML = "";
        prediction.top_predictions.forEach((guess, index) => {
            const card = document.createElement("article");
            card.className = "top-pick";
            card.innerHTML = `
                <strong>${guess.digit}</strong>
                <span>${index === 0 ? "Lead guess" : `Option ${index + 1}`}</span>
                <span>${formatPercent(guess.confidence)}</span>
            `;
            elements.topPredictions.appendChild(card);
        });

        elements.probabilitiesGrid.innerHTML = "";
        Object.entries(prediction.probabilities)
            .sort(([, left], [, right]) => right - left)
            .forEach(([digit, probability]) => {
                const row = document.createElement("div");
                row.className = "prob-item";
                row.innerHTML = `
                    <span class="prob-digit">${digit}</span>
                    <div class="prob-meter" aria-hidden="true"><span></span></div>
                    <span class="prob-value">${formatPercent(probability)}</span>
                `;
                row.querySelector(".prob-meter span").style.width = `${probability * 100}%`;
                elements.probabilitiesGrid.appendChild(row);
            });
    }

    async function fetchJson(url, options) {
        const response = await fetch(url, options);
        let payload;

        try {
            payload = await response.json();
        } catch (error) {
            throw new Error("The server returned an unreadable response.");
        }

        if (!response.ok) {
            throw new Error(payload.error || "The request did not complete successfully.");
        }

        return payload;
    }

    async function loadStats() {
        const stats = await fetchJson("/feedback_stats");
        elements.totalFeedback.textContent = stats.total_feedback;
        elements.correctionsCount.textContent = stats.wrong_predictions;
        elements.agreementRate.textContent = formatPercent(stats.agreement_rate);
        elements.averageConfidence.textContent = formatPercent(stats.average_confidence);

        if (stats.total_feedback === 0) {
            elements.retrainMeta.textContent =
                "No saved feedback yet. Retraining will use the reference dataset only.";
        } else {
            elements.retrainMeta.textContent =
                `${stats.total_feedback} saved feedback samples are ready for the next retraining run.`;
        }
    }

    async function loadHealth() {
        const health = await fetchJson("/health");

        if (health.retraining) {
            elements.healthBadge.textContent = "Retraining";
            elements.healthBadge.dataset.state = "busy";
            elements.healthDetail.textContent = "A model refresh is currently running.";
            return;
        }

        if (health.model_loaded) {
            elements.healthBadge.textContent = "Model ready";
            elements.healthBadge.dataset.state = "healthy";
            elements.healthDetail.textContent =
                `${health.feedback_count} feedback samples currently saved.`;
            return;
        }

        elements.healthBadge.textContent = "Model missing";
        elements.healthBadge.dataset.state = "warning";
        elements.healthDetail.textContent =
            health.model_error || "Run train_model.py to generate the initial model.";
    }

    async function loadStatus() {
        const results = await Promise.allSettled([loadStats(), loadHealth()]);
        const rejected = results.find((result) => result.status === "rejected");
        if (rejected) {
            throw rejected.reason;
        }
    }

    function paintCanvasBackground() {
        ctx.fillStyle = "#fffdf7";
        ctx.fillRect(0, 0, elements.canvas.width, elements.canvas.height);
    }

    function resetPredictionAfterCanvasChange() {
        if (!state.currentPrediction && !state.feedbackLocked) {
            return;
        }

        state.currentPrediction = null;
        state.feedbackLocked = false;
        closeCorrectionPanel();
        hideMessage(elements.feedbackSuccess);
        renderPrediction(null);
        updateFeedbackPrompt("Canvas changed. Run a fresh prediction when you're ready.");
    }

    function clearCanvas(initial) {
        paintCanvasBackground();
        state.isDrawing = false;
        state.hasInk = false;
        state.currentPrediction = null;
        state.feedbackLocked = false;
        state.currentImageData = elements.canvas.toDataURL("image/png");
        closeCorrectionPanel();
        renderPrediction(null);
        hideMessage(elements.errorMessage);

        if (initial) {
            updateFeedbackPrompt("Make a prediction to unlock feedback controls.");
            hideMessage(elements.feedbackSuccess);
        } else {
            updateFeedbackPrompt("Canvas cleared. Draw another digit to continue.");
            hideMessage(elements.feedbackSuccess);
        }

        elements.canvasHint.textContent = "Start drawing to unlock prediction.";
        refreshControlStates();
    }

    function getCanvasPoint(event) {
        const rect = elements.canvas.getBoundingClientRect();
        return {
            x: ((event.clientX - rect.left) * elements.canvas.width) / rect.width,
            y: ((event.clientY - rect.top) * elements.canvas.height) / rect.height,
        };
    }

    function startStroke(event) {
        if (state.busy) {
            return;
        }

        event.preventDefault();
        resetPredictionAfterCanvasChange();

        const point = getCanvasPoint(event);
        state.isDrawing = true;
        state.lastX = point.x;
        state.lastY = point.y;
        state.hasInk = true;

        ctx.fillStyle = "#15263a";
        ctx.beginPath();
        ctx.arc(point.x, point.y, ctx.lineWidth / 3, 0, Math.PI * 2);
        ctx.fill();

        state.currentImageData = elements.canvas.toDataURL("image/png");
        elements.canvasHint.textContent = "Nice start. Press Predict when the digit feels complete.";

        if (typeof event.pointerId === "number") {
            elements.canvas.setPointerCapture(event.pointerId);
        }

        refreshControlStates();
    }

    function drawStroke(event) {
        if (!state.isDrawing) {
            return;
        }

        event.preventDefault();
        const point = getCanvasPoint(event);

        ctx.beginPath();
        ctx.moveTo(state.lastX, state.lastY);
        ctx.lineTo(point.x, point.y);
        ctx.stroke();

        state.lastX = point.x;
        state.lastY = point.y;
        state.hasInk = true;
        state.currentImageData = elements.canvas.toDataURL("image/png");
        elements.canvasHint.textContent = "Looking good. Predict whenever you're ready.";
        refreshControlStates();
    }

    function endStroke(event) {
        if (!state.isDrawing) {
            return;
        }

        state.isDrawing = false;
        state.currentImageData = elements.canvas.toDataURL("image/png");

        if (event && typeof event.pointerId === "number") {
            try {
                elements.canvas.releasePointerCapture(event.pointerId);
            } catch (error) {
                // Ignore browsers that release capture automatically.
            }
        }
    }

    async function makePrediction() {
        if (!state.hasInk) {
            showError("Draw a digit before asking for a prediction.");
            return;
        }

        hideMessage(elements.errorMessage);
        hideMessage(elements.feedbackSuccess);
        closeCorrectionPanel();
        state.currentImageData = elements.canvas.toDataURL("image/png");

        setBusy(true, "Analyzing sketch...");

        try {
            const prediction = await fetchJson("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: state.currentImageData }),
            });

            state.currentPrediction = prediction;
            state.feedbackLocked = false;
            renderPrediction(prediction);
            updateFeedbackPrompt(`The model sees a ${prediction.digit}. Was it right?`);
            elements.canvasHint.textContent =
                "Want to refine the stroke? Draw again and predict once more.";
            await loadStatus();
        } catch (error) {
            showError(error.message);
        } finally {
            setBusy(false);
        }
    }

    async function submitFeedback(correctDigit) {
        if (!state.currentPrediction || state.feedbackLocked) {
            return;
        }

        hideMessage(elements.errorMessage);
        setBusy(
            true,
            correctDigit === state.currentPrediction.digit
                ? "Saving confirmation..."
                : "Saving correction..."
        );

        try {
            const result = await fetchJson("/feedback", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image: state.currentImageData,
                    predicted_digit: state.currentPrediction.digit,
                    correct_digit: correctDigit,
                    confidence: state.currentPrediction.confidence,
                }),
            });

            state.feedbackLocked = true;
            closeCorrectionPanel();

            if (result.was_correction) {
                updateFeedbackPrompt(
                    `Correction saved. The correct digit was ${correctDigit}.`
                );
            } else {
                updateFeedbackPrompt("Thanks for confirming that prediction.");
            }

            showMessage(
                elements.feedbackSuccess,
                `${result.total_entries} feedback samples saved so far.`
            );

            elements.canvasHint.textContent = "Draw another digit to keep collecting examples.";
            await loadStatus();
        } catch (error) {
            showError(error.message);
        } finally {
            setBusy(false);
        }
    }

    async function retrainModel() {
        hideMessage(elements.errorMessage);
        hideMessage(elements.feedbackSuccess);
        setBusy(true, "Retraining model...");

        try {
            const result = await fetchJson("/retrain", { method: "POST" });
            const summary = `Retrained with ${result.feedback_used} feedback samples. Held-out accuracy: ${formatPercent(result.test_accuracy)}.`;
            let retrainMessage;
            showMessage(elements.feedbackSuccess, summary);

            if (result.skipped_feedback > 0) {
                retrainMessage =
                    `${result.feedback_used} samples were used and ${result.skipped_feedback} were skipped because they could not be decoded cleanly.`;
            } else if (result.feedback_used > 0) {
                retrainMessage =
                    `${result.feedback_used} saved feedback samples were blended into the new training run.`;
            } else {
                retrainMessage =
                    "No saved feedback was available, so this run used the reference dataset only.";
            }

            await loadStatus();
            elements.retrainMeta.textContent = retrainMessage;
        } catch (error) {
            showError(error.message);
        } finally {
            setBusy(false);
        }
    }

    function toggleCorrectionPanel() {
        if (!state.currentPrediction || state.feedbackLocked || state.busy) {
            return;
        }

        if (state.correctionOpen) {
            closeCorrectionPanel();
            updateFeedbackPrompt(`The model sees a ${state.currentPrediction.digit}. Was it right?`);
        } else {
            openCorrectionPanel();
            updateFeedbackPrompt("Select the digit you actually drew.");
        }
    }

    function createDigitSelector() {
        for (let digit = 0; digit <= 9; digit += 1) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "digit-choice";
            button.textContent = String(digit);
            button.addEventListener("click", () => submitFeedback(digit));
            elements.digitSelector.appendChild(button);
        }
    }

    function bindEvents() {
        elements.canvas.addEventListener("pointerdown", startStroke);
        elements.canvas.addEventListener("pointermove", drawStroke);
        elements.canvas.addEventListener("pointerup", endStroke);
        elements.canvas.addEventListener("pointerleave", endStroke);
        elements.canvas.addEventListener("pointercancel", endStroke);
        elements.canvas.addEventListener("contextmenu", (event) => event.preventDefault());

        elements.clearBtn.addEventListener("click", () => clearCanvas(false));
        elements.predictBtn.addEventListener("click", makePrediction);
        elements.rightBtn.addEventListener("click", () => {
            if (state.currentPrediction) {
                submitFeedback(state.currentPrediction.digit);
            }
        });
        elements.wrongBtn.addEventListener("click", toggleCorrectionPanel);
        elements.retrainBtn.addEventListener("click", retrainModel);

        document.addEventListener("keydown", (event) => {
            if (event.repeat) {
                return;
            }

            if (event.key === "Enter") {
                makePrediction();
            } else if (event.key.toLowerCase() === "c") {
                clearCanvas(false);
            } else if (event.key.toLowerCase() === "r") {
                retrainModel();
            }
        });
    }

    function initCanvas() {
        ctx.lineWidth = 18;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.strokeStyle = "#15263a";
        paintCanvasBackground();
    }

    async function bootstrap() {
        createDigitSelector();
        initCanvas();
        bindEvents();
        clearCanvas(true);
        refreshControlStates();

        try {
            await loadStatus();
        } catch (error) {
            showError(error.message);
        }
    }

    bootstrap();
})();
