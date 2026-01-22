<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>QPCS AI Dashboard</title>
    <link rel="stylesheet" href="https://kendo.cdn.telerik.com/2022.3.1109/styles/kendo.default-v2.min.css"/>
    <script src="https://code.jquery.com/jquery-1.12.4.min.js"></script>
    <script src="https://kendo.cdn.telerik.com/2022.3.1109/js/kendo.all.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

    <style>
        /* UI ASLI ANDA (Original Style) */
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h2 { color: #203764; margin-bottom: 20px; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        
        /* TABS */
        .tabs { display: flex; margin-bottom: 20px; border-bottom: 1px solid #ddd; }
        .tab-btn { padding: 10px 20px; cursor: pointer; border: none; background: none; font-weight: 600; color: #666; font-size: 16px; border-bottom: 3px solid transparent; }
        .tab-btn.active { color: #4472C4; border-bottom: 3px solid #4472C4; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* FORM */
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
        .radio-group label { display: inline-block; margin-right: 20px; font-weight: normal; cursor: pointer; }
        
        /* BUTTONS */
        .btn-primary { background-color: #4472C4; color: white; border: none; padding: 12px 25px; border-radius: 4px; cursor: pointer; font-size: 16px; transition: 0.2s; display: inline-flex; align-items: center; gap: 8px; }
        .btn-primary:hover { background-color: #203764; }
        
        /* CONFIG BOX (CLEANSING) */
        .config-box { background: #f9fafb; padding: 15px; border-radius: 6px; border: 1px solid #e5e7eb; display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
        
        /* SWITCH TOGGLE (DISABLED STYLE) */
        .switch { position: relative; display: inline-block; width: 40px; height: 24px; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: not-allowed; top: 0; left: 0; right: 0; bottom: 0; background-color: #ccc; border-radius: 34px; }
        .slider:before { position: absolute; content: ""; height: 16px; width: 16px; left: 4px; bottom: 4px; background-color: #f1f1f1; border-radius: 50%; }
        
        /* MODAL LOADING */
        .modal-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.6); z-index: 1000; display: none; justify-content: center; align-items: center; }
        .modal-box { background: white; padding: 40px; border-radius: 12px; text-align: center; width: 400px; box-shadow: 0 10px 25px rgba(0,0,0,0.2); }
        .modal-icon { font-size: 50px; margin-bottom: 20px; }
        .fa-circle-notch { color: #4472C4; animation: spin 1s linear infinite; }
        .fa-check-circle { color: #00B050; display: none; }
        .fa-times-circle { color: #FF0000; display: none; }
        #modalTitle { font-size: 20px; font-weight: bold; margin-bottom: 10px; color: #333; }
        #modalSub { font-size: 14px; color: #666; line-height: 1.5; }
        #btnCloseModal { margin-top: 20px; background: #666; display: none; }

        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>

<div class="container">
    <h2><i class="fas fa-microchip"></i> QPCS AI Dashboard</h2>

    <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('predict')">Prediction (Usage)</button>
        <button class="tab-btn" onclick="switchTab('train')">Admin Training</button>
    </div>

    <div id="tab-predict" class="tab-content active">
        <div class="form-group">
            <label>1. Select Report Type:</label>
            <div class="radio-group">
                <label><input type="radio" name="reportType" value="daily" checked> Daily (Defect Only)</label>
                <label><input type="radio" name="reportType" value="monthly"> Monthly (Category + Defect)</label>
            </div>
        </div>

        <div class="form-group">
            <label>2. AI Configuration:</label>
            <div class="config-box">
                <div>
                    <div style="font-weight: 600; font-size:14px;">Deep Process Cleansing</div>
                    <div style="font-size:12px; color: #666;">Remove noise from data.</div>
                </div>
                <label class="switch">
                    <input type="checkbox" id="cleanPredict" disabled>
                    <span class="slider"></span>
                </label>
            </div>
            <div style="font-size:11px; color:#EF4444; margin-bottom:20px;">
                *Cleansing Disabled (RAW Data Mode)
            </div>
        </div>

        <div class="form-group">
            <label>3. Upload Excel File:</label>
            <input name="files" id="filePredict" type="file" aria-label="files" />
        </div>

        <button id="btnPredict" class="btn-primary">
            <i class="fas fa-play"></i> Run Prediction
        </button>
    </div>

    <div id="tab-train" class="tab-content">
        <div style="background: #e3f2fd; padding:15px; border-left:4px solid #2196f3; margin-bottom:20px; color:#0d47a1;">
            <i class="fas fa-info-circle"></i> 
            <b>Training Mode:</b> Upload data master history terbaru. Proses 5-10 menit.
        </div>

        <div class="form-group">
            <label>Upload Training Data (Master):</label>
            <input name="files" id="fileTrain" type="file" aria-label="files" />
        </div>

        <button id="btnTrain" class="btn-primary" style="background-color: #00B050;">
            <i class="fas fa-sync-alt"></i> Start Retraining
        </button>
    </div>
</div>

<div id="loadingModal" class="modal-overlay">
    <div class="modal-box">
        <i id="modalSpinner" class="fas fa-circle-notch modal-icon"></i>
        <i id="iconSuccess" class="fas fa-check-circle modal-icon"></i>
        <i id="iconError" class="fas fa-times-circle modal-icon"></i>
        
        <div id="modalTitle">Connecting...</div>
        <div id="modalSub">Initializing process...</div>
        
        <button id="btnCloseModal" class="btn-primary" onclick="$('#loadingModal').fadeOut();">Close</button>
    </div>
</div>

<script>
    // IP Address Server (Pastikan Benar)
    const API_BASE_URL = "http://13.229.172.201:8000";

    $(document).ready(function() {
        $("#filePredict").kendoUpload({ multiple: false, validation: { allowedExtensions: [".xlsx"] } });
        $("#fileTrain").kendoUpload({ multiple: false, validation: { allowedExtensions: [".xlsx"] } });
    });

    function switchTab(tabName) {
        $(".tab-content").removeClass("active");
        $(".tab-btn").removeClass("active");
        $("#tab-" + tabName).addClass("active");
        // Update button active state manually based on onclick text
        $(".tab-btn").each(function(){
            if($(this).attr("onclick").includes(tabName)) $(this).addClass("active");
        });
    }

    // ===============================================
    // POLLING ENGINE (Anti-Stuck Logic)
    // ===============================================
    let pollingInterval = null;

    function startPolling(processName) {
        if(pollingInterval) clearInterval(pollingInterval);
        
        pollingInterval = setInterval(function() {
            // Tambahkan timestamp (?t=) agar tidak kena Cache Browser
            $.ajax({
                url: API_BASE_URL + "/progress?t=" + new Date().getTime(),
                type: "GET",
                cache: false, 
                success: function(data) {
                    // Update Text Modal hanya jika ada data valid
                    if (data.is_running || data.progress > 0) {
                        $("#modalTitle").text(processName + ": " + data.progress + "%");
                        $("#modalSub").text(data.message);
                    }
                    
                    // Khusus Training: Stop otomatis jika 100%
                    if(processName === "Training") {
                        if(!data.is_running && data.progress == 100) {
                            stopPolling();
                            showResult("success", "Training Complete!", "AI updated.");
                        }
                        if(!data.is_running && data.message.includes("Error")) {
                            stopPolling();
                            showResult("error", "Failed", data.message);
                        }
                    }
                },
                error: function(e) {
                    console.log("Polling retry...", e);
                }
            });
        }, 1000); // Cek tiap 1 detik
    }

    function stopPolling() {
        if(pollingInterval) clearInterval(pollingInterval);
    }

    function showLoading(title, sub) {
        $("#loadingModal").fadeIn().css("display", "flex");
        $("#modalSpinner").show();
        $("#iconSuccess").hide();
        $("#iconError").hide();
        $("#btnCloseModal").hide();
        $("#modalTitle").text(title);
        $("#modalSub").text(sub);
    }

    function showResult(type, title, sub) {
        $("#modalSpinner").hide();
        $("#modalTitle").text(title);
        $("#modalSub").text(sub);
        $("#btnCloseModal").show();
        
        if(type === "success") $("#iconSuccess").show();
        else $("#iconError").show();
    }

    // ===============================================
    // CLICK HANDLER: PREDICT
    // ===============================================
    $("#btnPredict").click(function() {
        var upload = $("#filePredict").data("kendoUpload");
        var files = upload.getFiles();

        if (files.length === 0) { alert("Please select file!"); return; }

        var reportType = $("input[name='reportType']:checked").val();
        var formData = new FormData();
        formData.append("file", files[0].rawFile);

        showLoading("Connecting...", "Uploading file...");
        startPolling("Processing"); // Start polling immediately

        fetch(API_BASE_URL + "/predict?report_type=" + reportType + "&enable_cleansing=false", {
            method: "POST", 
            body: formData
        })
        .then(res => {
            if(!res.ok) return res.json().then(e => { throw new Error(e.detail || "Server Error") });
            return res.blob();
        })
        .then(blob => {
            stopPolling();
            showResult("success", "Success!", "Report Downloaded.");

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

    // ===============================================
    // CLICK HANDLER: TRAIN
    // ===============================================
    $("#btnTrain").click(function() {
        var upload = $("#fileTrain").data("kendoUpload");
        var files = upload.getFiles();

        if (files.length === 0) { alert("Please select file!"); return; }
        if (!confirm("Start Retraining?")) return;

        var formData = new FormData();
        formData.append("file", files[0].rawFile);

        showLoading("Initializing...", "Starting Training...");
        
        fetch(API_BASE_URL + "/train?enable_cleansing=false", {
            method: "POST", 
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            startPolling("Training");
        })
        .catch(err => {
            showResult("error", "Error", "Connection failed");
        });
    });

</script>

</body>
</html>