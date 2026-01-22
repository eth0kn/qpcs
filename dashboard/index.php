<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>QPCS AI Dashboard - Defect Classification</title>
    <link rel="stylesheet" href="https://kendo.cdn.telerik.com/2022.3.1109/styles/kendo.default-v2.min.css"/>
    <script src="https://code.jquery.com/jquery-1.12.4.min.js"></script>
    <script src="https://kendo.cdn.telerik.com/2022.3.1109/js/kendo.all.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

    <style>
        /* CSS SEDERHANA UNTUK LAYOUT */
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h2 { color: #203764; margin-bottom: 20px; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        
        /* TABS */
        .tabs { display: flex; margin-bottom: 20px; border-bottom: 1px solid #ddd; }
        .tab-btn { padding: 10px 20px; cursor: pointer; border: none; background: none; font-weight: 600; color: #666; font-size: 16px; border-bottom: 3px solid transparent; }
        .tab-btn.active { color: #4472C4; border-bottom: 3px solid #4472C4; }
        .tab-content { display: none; animation: fadeIn 0.3s; }
        .tab-content.active { display: block; }

        /* FORM ELEMENTS */
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
        .radio-group label { display: inline-block; margin-right: 20px; font-weight: normal; cursor: pointer; }
        
        /* BUTTONS */
        .btn-primary { background-color: #4472C4; color: white; border: none; padding: 12px 25px; border-radius: 4px; cursor: pointer; font-size: 16px; transition: 0.2s; display: inline-flex; align-items: center; gap: 8px; }
        .btn-primary:hover { background-color: #203764; }
        .btn-primary:disabled { background-color: #ccc; cursor: not-allowed; }

        /* CONFIG BOX (CLEANSING) */
        .config-box { background: #f9fafb; padding: 15px; border-radius: 6px; border: 1px solid #e5e7eb; display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        
        /* SWITCH TOGGLE */
        .switch { position: relative; display: inline-block; width: 40px; height: 24px; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #ccc; transition: .4s; border-radius: 34px; }
        .slider:before { position: absolute; content: ""; height: 16px; width: 16px; left: 4px; bottom: 4px; background-color: white; transition: .4s; border-radius: 50%; }
        input:checked + .slider { background-color: #4472C4; }
        input:checked + .slider:before { transform: translateX(16px); }
        /* Disabled Switch Style */
        input:disabled + .slider { background-color: #e0e0e0; cursor: not-allowed; }
        input:disabled + .slider:before { background-color: #bdbdbd; }

        /* MODAL STYLES */
        .modal-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.6); z-index: 1000; display: none; justify-content: center; align-items: center; }
        .modal-box { background: white; padding: 40px; border-radius: 12px; text-align: center; width: 400px; box-shadow: 0 10px 25px rgba(0,0,0,0.2); }
        .modal-icon { font-size: 50px; margin-bottom: 20px; }
        .fa-circle-notch { color: #4472C4; animation: spin 1s linear infinite; }
        .fa-check-circle { color: #00B050; display: none; }
        .fa-times-circle { color: #FF0000; display: none; }
        #modalTitle { font-size: 20px; font-weight: bold; margin-bottom: 10px; color: #333; }
        #modalSub { font-size: 14px; color: #666; line-height: 1.5; }
        #btnCloseModal { margin-top: 20px; background: #666; }

        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    </style>
</head>
<body>

<div class="container">
    <h2><i class="fas fa-robot"></i> QPCS AI Dashboard</h2>

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
            <div style="font-size:11px; color:#EF4444; margin-top:-15px; margin-bottom:20px;">
                *Cleansing dinonaktifkan (Mode RAW Data Mentah Aktif)
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
        <div class="alert" style="background: #eef2ff; color: #304ffe; padding: 15px; border-radius: 4px; margin-bottom: 20px; font-size: 14px;">
            <i class="fas fa-info-circle"></i> 
            <b>Training Mode:</b> Upload data master history terbaru untuk melatih kecerdasan AI.
            Proses ini akan memakan waktu 5-10 menit.
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
        
        <div id="modalTitle">Processing...</div>
        <div id="modalSub">Please wait while AI is analyzing your data.</div>
        
        <button id="btnCloseModal" class="btn-primary" style="display:none;" onclick="$('#loadingModal').fadeOut();">Close</button>
    </div>
</div>

<script>
    // KONFIGURASI API BACKEND (Ganti IP jika berubah)
    const API_BASE_URL = "http://13.229.172.201:8000";

    $(document).ready(function() {
        // Init Kendo Upload
        $("#filePredict").kendoUpload({ multiple: false });
        $("#fileTrain").kendoUpload({ multiple: false });
    });

    // Tab Switcher Logic
    function switchTab(tabName) {
        $(".tab-content").removeClass("active");
        $(".tab-btn").removeClass("active");
        $("#tab-" + tabName).addClass("active");
        // Update button style active
        $(".tab-btn").each(function() {
            if($(this).attr("onclick").includes(tabName)) {
                $(this).addClass("active");
            }
        });
    }

    // ==========================================================
    // LOGIC: PREDICTION (Dengan Real-time Polling)
    // ==========================================================
    $("#btnPredict").click(function() {
        var upload = $("#filePredict").data("kendoUpload");
        var files = upload.getFiles();

        if (files.length === 0) {
            alert("Please select a file first!");
            return;
        }

        var reportType = $("input[name='reportType']:checked").val();
        var isClean = "false"; // Force RAW

        var formData = new FormData();
        formData.append("file", files[0].rawFile);

        // 1. Reset & Show Modal UI
        $("#loadingModal").css("display", "flex").hide().fadeIn();
        $("#modalSpinner").show();
        $("#iconSuccess").hide();
        $("#iconError").hide();
        $("#btnCloseModal").hide();
        
        $("#modalTitle").text("Connecting...");
        $("#modalSub").text("Initializing upload...");

        // 2. Setup Polling (Cek progress setiap 1 detik)
        let progressInterval = null;
        
        function startPolling() {
            if (progressInterval) clearInterval(progressInterval);
            progressInterval = setInterval(function() {
                // Tambahkan timestamp (?t=...) agar browser tidak cache JSON
                $.ajax({
                    url: API_BASE_URL + "/progress?t=" + new Date().getTime(),
                    type: "GET",
                    success: function(data) {
                        console.log("Progress:", data);
                        if (data.is_running) {
                            // Update Teks Modal Real-time
                            $("#modalTitle").text("Processing: " + data.progress + "%");
                            $("#modalSub").text(data.message);
                        }
                    },
                    error: function(e) {
                        console.log("Polling silent error:", e);
                    }
                });
            }, 1000);
        }

        startPolling(); // Mulai polling segera

        // 3. Kirim Request Utama (Fetch)
        fetch(API_BASE_URL + "/predict?report_type=" + reportType + "&enable_cleansing=" + isClean, {
            method: "POST", 
            body: formData
        })
        .then(res => {
            if(!res.ok) {
                return res.json().then(err => { throw new Error(err.detail || "Server Error"); });
            }
            return res.blob(); // Menghapkan file excel
        })
        .then(blob => {
            clearInterval(progressInterval); // Stop polling jika sukses
            
            // Update UI Sukses
            $("#modalSpinner").hide(); 
            $("#iconSuccess").fadeIn();
            $("#modalTitle").text("Success!");
            $("#modalSub").text("Analysis Complete. Report Downloaded.");
            $("#btnCloseModal").show();

            // Trigger Download Browser
            var url = window.URL.createObjectURL(blob);
            var a = document.createElement("a"); 
            a.href = url;
            a.download = "RESULT_" + files[0].name;
            document.body.appendChild(a); 
            a.click(); 
            a.remove();
        })
        .catch(err => {
            clearInterval(progressInterval); // Stop polling jika error
            
            // Update UI Gagal
            $("#modalSpinner").hide(); 
            $("#iconError").fadeIn();
            $("#modalTitle").text("Failed"); 
            $("#modalSub").text(err.message);
            $("#btnCloseModal").show();
        });
    });

    // ==========================================================
    // LOGIC: TRAINING (Sama-sama Real-time)
    // ==========================================================
    $("#btnTrain").click(function() {
        var upload = $("#fileTrain").data("kendoUpload");
        var files = upload.getFiles();

        if (files.length === 0) {
            alert("Please select training master file!");
            return;
        }

        if(!confirm("Are you sure want to retrain the AI? This will take 5-10 minutes.")) return;

        var formData = new FormData();
        formData.append("file", files[0].rawFile);

        $("#loadingModal").css("display", "flex").hide().fadeIn();
        $("#modalSpinner").show();
        $("#iconSuccess").hide();
        $("#iconError").hide();
        $("#btnCloseModal").hide();
        
        $("#modalTitle").text("Training Started");
        $("#modalSub").text("Uploading dataset...");

        // Polling Logic
        let trainInterval = setInterval(function() {
            $.ajax({
                url: API_BASE_URL + "/progress?t=" + new Date().getTime(),
                type: "GET",
                success: function(data) {
                    if (data.is_running) {
                        $("#modalTitle").text("Training: " + data.progress + "%");
                        $("#modalSub").text(data.message);
                    }
                }
            });
        }, 1500);

        // Fetch Request
        fetch(API_BASE_URL + "/train?enable_cleansing=false", {
            method: "POST", 
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            // Training backend berjalan di background thread (async), 
            // jadi fetch akan return 'started' dgn cepat.
            // Kita lanjutkan polling sampai status backend idle/finish.
            
            // (Optional logic: bisa handle 'started' message disini)
        })
        .catch(err => {
            clearInterval(trainInterval);
            $("#modalSpinner").hide(); 
            $("#iconError").fadeIn();
            $("#modalTitle").text("Error"); 
            $("#modalSub").text("Failed to connect server.");
            $("#btnCloseModal").show();
        });
        
        // Khusus Training: Karena backend jalan di thread, kita pantau terus sampai selesai
        // Logic tambahan untuk stop interval training
        let checkFinish = setInterval(function(){
             $.ajax({
                url: API_BASE_URL + "/progress?t=" + new Date().getTime(),
                type: "GET",
                success: function(data) {
                    // Jika progress 100 atau is_running false tapi progress > 0
                    if (!data.is_running && data.progress == 100) {
                        clearInterval(trainInterval);
                        clearInterval(checkFinish);
                        
                        $("#modalSpinner").hide(); 
                        $("#iconSuccess").fadeIn();
                        $("#modalTitle").text("Training Complete!");
                        $("#modalSub").text("AI has been updated.");
                        $("#btnCloseModal").show();
                    }
                    else if (!data.is_running && data.progress == 0 && data.message.includes("Error")) {
                        clearInterval(trainInterval);
                        clearInterval(checkFinish);
                        
                        $("#modalSpinner").hide(); 
                        $("#iconError").fadeIn();
                        $("#modalTitle").text("Training Failed");
                        $("#modalSub").text(data.message);
                        $("#btnCloseModal").show();
                    }
                }
            });
        }, 2000);
    });

</script>

</body>
</html>