<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QPCS AI Admin Dashboard</title>
    
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://kendo.cdn.telerik.com/2023.1.117/styles/kendo.common.min.css" />
    <link rel="stylesheet" href="https://kendo.cdn.telerik.com/2023.1.117/styles/kendo.default.min.css" />
    <script src="https://code.jquery.com/jquery-3.6.1.min.js"></script>
    <script src="https://kendo.cdn.telerik.com/2023.1.117/js/kendo.all.min.js"></script>

    <style>
        :root { --primary: #4F46E5; --primary-hover: #4338ca; --bg-surface: #ffffff; --bg-body: #F3F4F6; --text-main: #111827; --text-muted: #6B7280; --border-color: #E5E7EB; }
        body { font-family: 'Inter', sans-serif; background-color: var(--bg-body); margin: 0; padding: 20px; color: var(--text-main); }
        .container { max-width: 1200px; margin: 0 auto; display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        .card { background: var(--bg-surface); border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 24px; border: 1px solid var(--border-color); }
        .card-header { margin-bottom: 20px; border-bottom: 1px solid var(--border-color); padding-bottom: 15px; }
        .card-title { font-size: 1.25rem; font-weight: 600; color: var(--text-main); margin: 0; display: flex; align-items: center; gap: 10px; }
        .card-icon { background: #EEF2FF; color: var(--primary); padding: 8px; border-radius: 8px; }
        .form-group { margin-bottom: 16px; }
        .form-label { display: block; font-weight: 500; margin-bottom: 8px; color: var(--text-main); }
        .btn-primary { background-color: var(--primary); color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 500; cursor: pointer; width: 100%; transition: all 0.2s; }
        .btn-primary:hover { background-color: var(--primary-hover); }
        .status-text-active { font-weight: 600; color: var(--primary); animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
        .modal-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); display: none; justify-content: center; align-items: center; z-index: 1000; backdrop-filter: blur(4px); }
        .modal-content { background: white; padding: 30px; border-radius: 16px; text-align: center; width: 400px; box-shadow: 0 10px 25px rgba(0,0,0,0.2); }
        .loader { border: 4px solid #f3f3f3; border-top: 4px solid var(--primary); border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 15px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>

<div class="container">
    <div class="card">
        <div class="card-header"><h2 class="card-title"><span class="card-icon">⚡</span> AI Prediction</h2></div>
        <div class="form-group"><label class="form-label">Upload Raw Data (Excel)</label><input type="file" id="fileInput" /></div>
        <div class="form-group"><label class="form-label">Report Type</label><select id="reportType" style="width: 100%;"><option value="daily">Daily Report (Defect)</option><option value="monthly">Monthly Report (Category)</option></select></div>
        <div class="form-group"><label class="form-label">Options</label><div style="display: flex; align-items: center; gap: 8px;">
            <input type="checkbox" id="chkCleanPredict"><label for="chkCleanPredict" style="margin:0; color: var(--text-muted);">Enable Data Cleansing (AI Pre-processing)</label>
        </div></div>
        <button class="btn-primary" id="btnProcess">Start Prediction</button>
    </div>

    <div class="card">
        <div class="card-header"><h2 class="card-title"><span class="card-icon">🧠</span> Retrain Model</h2></div>
        <div class="form-group"><label class="form-label">Upload Training Dataset (Excel)</label><input type="file" id="trainFileInput" /><p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 5px;">Supports: 'PROCESS (DEFECT)' and 'PROCESS (OZ,MS,IH)' sheets.</p></div>
        <div class="form-group"><label class="form-label">Options</label><div style="display: flex; align-items: center; gap: 8px;">
            <input type="checkbox" id="chkCleanTrain"><label for="chkCleanTrain" style="margin:0; color: var(--text-muted);">Enable Data Cleansing</label>
        </div></div>
        <button class="btn-primary" id="btnTrain" style="background-color: #10B981;">Start Training</button>
    </div>
</div>

<div class="modal-overlay" id="processModal">
    <div class="modal-content">
        <div id="modalSpinner" class="loader"></div>
        <div id="iconSuccess" style="display:none; color: #10B981; font-size: 3rem; margin-bottom: 15px;">✓</div>
        <div id="iconError" style="display:none; color: #EF4444; font-size: 3rem; margin-bottom: 15px;">✕</div>
        <h3 id="modalTitle" style="margin: 0 0 10px; font-weight: 600;">Processing...</h3>
        <p id="modalSub" style="color: var(--text-muted); margin: 0 0 20px;">Please wait while AI is working.</p>
        <div id="progressWrapper" style="width: 100%; background-color: #e5e7eb; border-radius: 9999px; height: 8px; margin-bottom: 20px; display: none;">
             <div id="progressBar" style="background-color: var(--primary); height: 8px; border-radius: 9999px; width: 0%; transition: width 0.5s;"></div>
        </div>
        <button id="btnCloseModal" class="btn-primary" style="display:none; width: auto; padding: 8px 25px;">Close</button>
    </div>
</div>

<script>
    $(document).ready(function() {
        $("#fileInput").kendoUpload({ multiple: false });
        $("#trainFileInput").kendoUpload({ multiple: false });
        $("#reportType").kendoDropDownList();

        // BIARKAN KOSONG (Relative Path)
        const API_URL = ""; 

        function showModal(title, sub) {
            $("#processModal").css("display", "flex");
            $("#modalTitle").text(title);
            $("#modalSub").text(sub).removeClass("status-text-active");
            $("#modalSpinner").show();
            $("#iconSuccess, #iconError, #btnCloseModal, #progressWrapper").hide();
            $("#progressBar").css("width", "0%");
        }
        $("#btnCloseModal").click(function() { $("#processModal").fadeOut(); });

        // --- FIX PADA BAGIAN INI ---
        $("#btnProcess").click(function() {
            var files = $("#fileInput").data("kendoUpload").getFiles();
            if (files.length === 0) { alert("Please select a file first."); return; }
            showModal("AI Prediction", "Reading file and analyzing...");

            // 1. Siapkan File dalam Body
            var formData = new FormData();
            formData.append("file", files[0].rawFile);

            // 2. Siapkan Parameter di URL (Query String) -> SOLUSI 422 ERROR
            var reportType = $("#reportType").val();
            var doClean = $("#chkCleanPredict").is(":checked");
            
            // Gabungkan URL
            var finalUrl = API_URL + "/predict?report_type=" + reportType + "&enable_cleansing=" + doClean;

            $.ajax({
                url: finalUrl, // Gunakan URL yang sudah ada parameternya
                type: "POST", 
                data: formData, 
                processData: false, 
                contentType: false, 
                xhrFields: { responseType: 'blob' },
                success: function(blob, status, xhr) {
                    $("#modalSpinner").hide(); $("#iconSuccess").fadeIn(); $("#modalTitle").text("Success!"); $("#modalSub").text("Download starting..."); $("#btnCloseModal").show();
                    
                    // Clear Files
                    $(".k-upload-files").remove(); $(".k-upload-status").remove();
                    $(".k-upload.k-header").addClass("k-upload-empty");
                    $(".k-upload-button").removeClass("k-state-focused");
                    $("#fileInput").data("kendoUpload").clearAllFiles();

                    var filename = ""; var disposition = xhr.getResponseHeader('Content-Disposition');
                    if (disposition && disposition.indexOf('attachment') !== -1) { var matches = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition); if (matches != null && matches[1]) filename = matches[1].replace(/['"]/g, ''); }
                    var link = document.createElement('a'); link.href = window.URL.createObjectURL(blob); link.download = filename || "Result.xlsx"; link.click();
                },
                error: function(xhr) { 
                    $("#modalSpinner").hide(); $("#iconError").fadeIn(); $("#modalTitle").text("Error"); 
                    // Show detailed error if available
                    var msg = "Something went wrong.";
                    if(xhr.responseText) { 
                        try { 
                            // Coba baca pesan error JSON dari FastAPI (misal: Validation Error)
                            // Karena responseType blob, kita perlu baca blob sebagai text dulu (agak tricky di jquery simple),
                            // tapi biasanya status code 422 sudah cukup jelas.
                            msg = "Validation Error (422). Check inputs.";
                        } catch(e){} 
                    }
                    $("#modalSub").text(msg); 
                    $("#btnCloseModal").show(); 
                }
            });
        });

        // TRAINING
        $("#btnTrain").click(function() {
            var files = $("#trainFileInput").data("kendoUpload").getFiles();
            if (files.length === 0) { alert("Please select a dataset file."); return; }
            showModal("Initializing Training", "Uploading dataset..."); $("#progressWrapper").show();

            var formData = new FormData();
            formData.append("file", files[0].rawFile);
            
            // Parameter juga masuk ke URL di sini
            var doClean = $("#chkCleanTrain").is(":checked");
            var finalUrl = API_URL + "/train?enable_cleansing=" + doClean;

            fetch(finalUrl, { method: "POST", body: formData })
            .then(response => {
                if(response.ok) {
                    let poller = setInterval(() => {
                        $.get(API_URL + "/train/status", function(data) {
                            $("#progressBar").css("width", data.progress + "%");
                            if(data.is_running) $("#modalSub").addClass("status-text-active").text(data.message);
                            if (!data.is_running && data.progress === 100) {
                                clearInterval(poller);
                                $("#modalSpinner").hide(); $("#iconSuccess").fadeIn(); $("#modalTitle").text("Training Complete!"); $("#modalSub").removeClass("status-text-active").text("New models ready."); $("#btnCloseModal").show();
                                
                                $(".k-upload-files").remove(); $(".k-upload-status").remove();
                                $(".k-upload.k-header").addClass("k-upload-empty");
                                $(".k-upload-button").removeClass("k-state-focused");
                                $("#trainFileInput").data("kendoUpload").clearAllFiles();
                            }
                            if(data.message && data.message.startsWith("Error")) {
                                clearInterval(poller); $("#modalSpinner").hide(); $("#iconError").fadeIn(); $("#modalTitle").text("Training Failed"); $("#modalSub").removeClass("status-text-active").text(data.message); $("#btnCloseModal").show();
                            }
                        });
                    }, 1000);
                } else { throw new Error("Failed to start training."); }
            }).catch(err => { $("#modalSpinner").hide(); $("#iconError").fadeIn(); $("#modalTitle").text("Error"); $("#modalSub").text(err.message); $("#btnCloseModal").show(); });
        });
    });
</script>
</body>
</html>