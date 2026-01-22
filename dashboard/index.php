<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QPCS AI Dashboard</title>
    
    <link rel="stylesheet" href="https://kendo.cdn.telerik.com/2022.3.1109/styles/kendo.default-v2.min.css"/>
    <script src="https://code.jquery.com/jquery-1.12.4.min.js"></script>
    <script src="https://kendo.cdn.telerik.com/2022.3.1109/js/kendo.all.min.js"></script>
    
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

    <style>
        /* BASE STYLES */
        body { font-family: 'Segoe UI', Arial, sans-serif; background-color: #f0f2f5; margin: 0; padding: 20px; }
        
        .main-card {
            background: #fff;
            max-width: 900px;
            margin: 0 auto;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow: hidden;
            border-top: 5px solid #0058e9; /* Kendo Blue */
        }

        .header-title {
            padding: 20px;
            border-bottom: 1px solid #e6e6e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header-title h2 { margin: 0; color: #333; font-size: 20px; font-weight: 600; }
        .status-badge { background: #e3fcef; color: #00a65a; padding: 5px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; border: 1px solid #00a65a; }

        .content-body { padding: 30px; }
        
        /* FORM ELEMENTS */
        .form-section { margin-bottom: 25px; padding-bottom: 25px; border-bottom: 1px solid #f0f0f0; }
        .form-label { display: block; font-weight: 600; color: #444; margin-bottom: 10px; }
        
        /* CUSTOM SWITCH (DISABLED) */
        .switch-wrapper { display: flex; align-items: center; justify-content: space-between; background: #f9f9f9; padding: 10px; border: 1px solid #e0e0e0; border-radius: 4px; }
        .switch-disabled { opacity: 0.6; cursor: not-allowed; }

        /* KENDO OVERRIDES FOR MOBILE */
        .k-upload .k-dropzone { padding: 10px; }
        
        /* MODAL OVERLAY (Custom styled to look like Kendo Window) */
        #progressOverlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.5); z-index: 9999;
            display: none; justify-content: center; align-items: center;
        }
        .progress-box {
            background: white; width: 90%; max-width: 400px;
            border-radius: 4px; box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            text-align: center; padding: 30px;
            border-top: 4px solid #ff6358; /* Kendo Red/Orange accent */
        }
        .progress-spinner { font-size: 40px; color: #0058e9; margin-bottom: 15px; }
        .progress-title { font-size: 18px; font-weight: bold; color: #333; margin-bottom: 5px; }
        .progress-detail { font-size: 14px; color: #666; margin-bottom: 20px; min-height: 20px; }
        
        .k-button-success { background-color: #00B050 !important; border-color: #00B050 !important; color: white !important; }
        .k-button-primary { background-color: #0058e9 !important; border-color: #0058e9 !important; color: white !important; }

        /* Responsive */
        @media (max-width: 600px) {
            .content-body { padding: 15px; }
            .k-tabstrip-items .k-item { display: block; width: 100%; }
        }
    </style>
</head>
<body>

<div class="main-card">
    <div class="header-title">
        <h2><i class="fas fa-microchip"></i> QPCS AI Dashboard</h2>
        <span class="status-badge">SYSTEM ONLINE</span>
    </div>

    <div id="tabstrip">
        <ul>
            <li class="k-state-active">Prediction (User)</li>
            <li>Admin Training</li>
        </ul>

        <div class="content-body">
            <div class="form-section">
                <span class="form-label">1. Report Type</span>
                <ul id="reportType" style="list-style:none; padding:0;">
                    <li style="margin-bottom:5px;">
                        <input type="radio" name="rpt" id="daily" value="daily" checked class="k-radio">
                        <label class="k-radio-label" for="daily">Daily Report (Defect Only)</label>
                    </li>
                    <li>
                        <input type="radio" name="rpt" id="monthly" value="monthly" class="k-radio">
                        <label class="k-radio-label" for="monthly">Monthly Report (Category + Defect)</label>
                    </li>
                </ul>
            </div>

            <div class="form-section">
                <span class="form-label">2. AI Configuration</span>
                <div class="switch-wrapper switch-disabled">
                    <div>
                        <strong>Deep Cleansing</strong><br>
                        <small style="color:#666">Remove noise & invalid characters</small>
                    </div>
                    <input type="checkbox" id="cleanSwitch" disabled checked="checked" />
                </div>
                <div style="font-size:11px; color:red; margin-top:5px;">* Cleansing Disabled (RAW Data Mode)</div>
            </div>

            <div class="form-section">
                <span class="form-label">3. Upload Data (.xlsx)</span>
                <input name="files" id="filePredict" type="file" aria-label="files" />
            </div>

            <button id="btnPredict" class="k-button k-button-lg k-button-solid-primary k-button-primary" style="width:100%; padding:15px;">
                <i class="fas fa-play" style="margin-right:8px;"></i> RUN PREDICTION
            </button>
        </div>

        <div class="content-body">
            <div class="k-messagebox k-messagebox-info" style="background:#e3f2fd; padding:15px; border-left:4px solid #2196f3; margin-bottom:20px;">
                <strong><i class="fas fa-info-circle"></i> Info:</strong> 
                Upload Master Data history terbaru untuk melatih ulang AI. Proses memakan waktu 5-10 menit.
            </div>

            <div class="form-section">
                <span class="form-label">Master Dataset (.xlsx)</span>
                <input name="files" id="fileTrain" type="file" aria-label="files" />
            </div>

            <button id="btnTrain" class="k-button k-button-lg k-button-solid-primary k-button-success" style="width:100%; padding:15px;">
                <i class="fas fa-sync-alt" style="margin-right:8px;"></i> START RETRAINING
            </button>
        </div>
    </div>
</div>

<div id="progressOverlay">
    <div class="progress-box">
        <i id="pIcon" class="fas fa-circle-notch fa-spin progress-spinner"></i>
        <div id="pTitle" class="progress-title">Connecting...</div>
        <div id="pSub" class="progress-detail">Initializing...</div>
        
        <button id="btnCloseOverlay" class="k-button" style="display:none; margin-top:15px;" onclick="$('#progressOverlay').fadeOut()">Close</button>
    </div>
</div>

<script>
    // KONFIGURASI IP BACKEND
    const API_BASE_URL = "http://13.229.172.201:8000";

    $(document).ready(function() {
        // Init Kendo Widgets
        $("#tabstrip").kendoTabStrip({ animation:  { open: { effects: "fadeIn" } } });
        
        $("#filePredict").kendoUpload({ multiple: false, validation: { allowedExtensions: [".xlsx"] } });
        $("#fileTrain").kendoUpload({ multiple: false, validation: { allowedExtensions: [".xlsx"] } });
        
        $("#cleanSwitch").kendoSwitch({ checked: false, enabled: false });

        // ============================================
        // LOGIC: POLLING ENGINE (Anti-Stuck)
        // ============================================
        let pollingId = null;

        function startProgressPolling(mode) {
            if (pollingId) clearInterval(pollingId);
            
            // Poll setiap 1 detik
            pollingId = setInterval(function() {
                // Tambahkan Timestamp (?t=) agar browser tidak cache hasil lama
                $.ajax({
                    url: API_BASE_URL + "/progress?t=" + new Date().getTime(),
                    type: "GET",
                    cache: false, // Force JQuery no-cache
                    success: function(data) {
                        console.log("Poll:", data);
                        
                        if (data.is_running || data.progress > 0) {
                            $("#pTitle").text(mode + ": " + data.progress + "%");
                            $("#pSub").text(data.message);
                        }
                        
                        // Handle Training Completion Logic via Polling
                        if (mode === "Training") {
                            if (!data.is_running && data.progress === 100) {
                                stopPolling();
                                showResult("success", "Training Complete!", "AI has been updated.");
                            }
                            if (!data.is_running && data.message.includes("Error")) {
                                stopPolling();
                                showResult("error", "Training Failed", data.message);
                            }
                        }
                    },
                    error: function(err) {
                        console.log("Polling error (Retrying...):", err);
                    }
                });
            }, 1000);
        }

        function stopPolling() {
            if (pollingId) clearInterval(pollingId);
        }

        function showLoading(title, sub) {
            $("#progressOverlay").fadeIn().css("display", "flex");
            $("#pIcon").attr("class", "fas fa-circle-notch fa-spin progress-spinner").css("color", "#0058e9");
            $("#pTitle").text(title);
            $("#pSub").text(sub);
            $("#btnCloseOverlay").hide();
        }

        function showResult(type, title, sub) {
            $("#pTitle").text(title);
            $("#pSub").text(sub);
            $("#btnCloseOverlay").show();
            
            if (type === "success") {
                $("#pIcon").attr("class", "fas fa-check-circle").css("color", "#00B050");
            } else {
                $("#pIcon").attr("class", "fas fa-times-circle").css("color", "#FF0000");
            }
        }

        // ============================================
        // EVENT: PREDICT CLICK
        // ============================================
        $("#btnPredict").click(function() {
            var upload = $("#filePredict").data("kendoUpload");
            var files = upload.getFiles();

            if (files.length === 0) {
                alert("Please upload a file first!");
                return;
            }

            var reportType = $("input[name='rpt']:checked").val();
            var formData = new FormData();
            formData.append("file", files[0].rawFile);

            showLoading("Connecting...", "Uploading file...");
            startProgressPolling("Processing"); // Start polling immediately

            // Fetch Request
            fetch(API_BASE_URL + "/predict?report_type=" + reportType + "&enable_cleansing=false", {
                method: "POST",
                body: formData
            })
            .then(res => {
                if (!res.ok) return res.json().then(e => { throw new Error(e.detail || "Server Error") });
                return res.blob();
            })
            .then(blob => {
                stopPolling();
                showResult("success", "Success!", "Downloading Report...");
                
                // Trigger Download
                var url = window.URL.createObjectURL(blob);
                var a = document.createElement("a");
                a.href = url;
                a.download = "RESULT_" + files[0].name;
                document.body.appendChild(a);
                a.click();
                a.remove();
            })
            .catch(err => {
                stopPolling();
                showResult("error", "Failed", err.message);
            });
        });

        // ============================================
        // EVENT: TRAIN CLICK
        // ============================================
        $("#btnTrain").click(function() {
            var upload = $("#fileTrain").data("kendoUpload");
            var files = upload.getFiles();

            if (files.length === 0) {
                alert("Please upload Master Data first!");
                return;
            }

            if (!confirm("Start AI Retraining? This will take 5-10 minutes.")) return;

            var formData = new FormData();
            formData.append("file", files[0].rawFile);

            showLoading("Initializing...", "Starting Training Process...");
            
            // Panggil Endpoint Train
            fetch(API_BASE_URL + "/train?enable_cleansing=false", {
                method: "POST",
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                // Training berjalan di background thread (async)
                // Kita mulai polling untuk memantau progressnya
                startProgressPolling("Training");
            })
            .catch(err => {
                showResult("error", "Connection Error", "Cannot reach server.");
            });
        });

    });
</script>

</body>
</html>