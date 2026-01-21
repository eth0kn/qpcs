<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QPCS AI Admin Dashboard</title>
    
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    
    <link rel="stylesheet" href="https://kendo.cdn.telerik.com/2023.1.117/styles/kendo.common.min.css" />
    <link rel="stylesheet" href="https://kendo.cdn.telerik.com/2023.1.117/styles/kendo.default.min.css" />
    <link rel="stylesheet" href="https://kendo.cdn.telerik.com/2023.1.117/styles/kendo.default.mobile.min.css" />
    <script src="https://code.jquery.com/jquery-3.6.1.min.js"></script>
    <script src="https://kendo.cdn.telerik.com/2023.1.117/js/kendo.all.min.js"></script>

    <style>
        :root {
            --primary: #4F46E5; /* Indigo Modern */
            --primary-hover: #4338ca;
            --bg-body: #F3F4F6;
            --card-bg: #FFFFFF;
            --text-main: #1F2937;
            --text-sub: #6B7280;
            --border: #E5E7EB;
            --success: #10B981;
            --danger: #EF4444;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-body);
            color: var(--text-main);
            margin: 0; padding: 20px;
            display: flex; justify-content: center; min-height: 100vh;
        }

        .main-wrapper {
            width: 100%; max-width: 750px;
            background: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.05);
            padding: 30px; margin: auto;
        }

        /* HEADER */
        header { text-align: center; margin-bottom: 20px; }
        h1 { margin: 0; font-size: 24px; font-weight: 700; color: #111; letter-spacing: -0.5px; }
        .subtitle { color: var(--text-sub); font-size: 14px; margin-top: 5px; }

        /* PRIVACY BADGE (RESTORED) */
        .privacy-badge {
            background-color: #EFF6FF;
            border: 1px solid #DBEAFE;
            color: #1E40AF;
            padding: 12px;
            border-radius: 8px;
            font-size: 13px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin-bottom: 25px;
        }
        .privacy-badge svg { width: 16px; height: 16px; fill: currentColor; }

        /* TABS NAVIGATION */
        .tabs { display: flex; border-bottom: 2px solid #E5E7EB; margin-bottom: 30px; }
        .tab-btn {
            flex: 1; text-align: center; padding: 15px; cursor: pointer;
            font-weight: 600; color: #9CA3AF; border-bottom: 2px solid transparent;
            margin-bottom: -2px; transition: all 0.3s;
        }
        .tab-btn:hover { color: var(--primary); background: #F9FAFB; }
        .tab-btn.active { color: var(--primary); border-bottom-color: var(--primary); }
        
        .tab-content { display: none; animation: fadeIn 0.4s; }
        .tab-content.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }

        /* SECTIONS & CARDS */
        .section-label {
            font-size: 13px; font-weight: 700; color: var(--text-sub);
            margin-bottom: 10px; display: block; text-transform: uppercase; letter-spacing: 0.5px;
        }

        .report-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 25px; }

        .report-card {
            border: 2px solid var(--border); border-radius: 12px; padding: 15px;
            cursor: pointer; transition: all 0.2s ease; position: relative;
        }
        .report-card:hover { border-color: var(--primary); background: #EEF2FF; }
        .report-card.active { border-color: var(--primary); background-color: #EEF2FF; box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.1); }
        .report-card input { position: absolute; opacity: 0; }

        .card-title { font-weight: 700; display: block; font-size: 15px; margin-bottom: 4px; }
        .card-desc { font-size: 12px; color: var(--text-sub); }
        .check-icon { position: absolute; top: 12px; right: 12px; color: var(--primary); display: none; }
        .report-card.active .check-icon { display: block; }

        /* CONFIG BOX (TOGGLE) */
        .config-box {
            background: #F9FAFB; border: 1px solid var(--border); border-radius: 12px;
            padding: 15px 20px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 25px;
        }
        
        /* SWITCH STYLE */
        .switch { position: relative; display: inline-block; width: 46px; height: 24px; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #D1D5DB; transition: .4s; border-radius: 34px; }
        .slider:before { position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px; background-color: white; transition: .4s; border-radius: 50%; }
        input:checked + .slider { background-color: var(--success); }
        input:checked + .slider:before { transform: translateX(22px); }

        /* BUTTONS */
        .btn-process {
            width: 100%; background-color: var(--primary); color: white; border: none;
            padding: 16px; border-radius: 12px; font-size: 15px; font-weight: 600;
            cursor: pointer; transition: background 0.3s; margin-top: 20px;
            display: flex; align-items: center; justify-content: center; gap: 8px;
        }
        .btn-process:hover { background-color: var(--primary-hover); }
        .btn-process:disabled { background-color: #E5E7EB; color: #9CA3AF; cursor: not-allowed; }

        .btn-train { background-color: #111827; } /* Dark button for Admin */
        .btn-train:hover { background-color: #000000; }

        /* UPLOAD OVERRIDE */
        .k-upload { border-radius: 12px !important; border-color: var(--border) !important; }
        .k-dropzone { background: #fff !important; padding: 15px !important; }

        /* --- MODAL LOADING (Simple & Elegant) --- */
        .modal-overlay {
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(5px);
            z-index: 9999; align-items: center; justify-content: center; flex-direction: column;
        }
        .loading-box {
            background: white; padding: 40px; border-radius: 20px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.15);
            text-align: center; width: 320px;
        }
        .spinner {
            width: 40px; height: 40px; border: 4px solid #E5E7EB;
            border-top: 4px solid var(--primary); border-radius: 50%;
            margin: 0 auto 20px auto; animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .loading-text { font-weight: 600; color: var(--text-main); margin-bottom: 5px; }
        .loading-sub { font-size: 13px; color: var(--text-sub); }
        
        .modal-icon { font-size: 40px; margin-bottom: 15px; display: none; }
        .btn-close-modal {
            margin-top: 20px; background: #F3F4F6; border: none; padding: 8px 20px;
            border-radius: 8px; cursor: pointer; display: none; font-weight: 500;
        }
        .btn-close-modal:hover { background: #E5E7EB; }

        /* Responsive */
        @media (max-width: 600px) {
            .report-grid { grid-template-columns: 1fr; }
            .main-wrapper { padding: 20px; }
        }
    </style>
</head>
<body>

<div class="main-wrapper">
    <header>
        <h1>QPCS AI Dashboard</h1>
        <div class="subtitle">Intelligent Defect & Category Classification</div>
    </header>

    <div class="privacy-badge">
        <span style="font-size:18px;">🔒</span>
        <div>
            <strong>Zero Persistence Guarantee:</strong><br>
            All uploaded data is processed in RAM and deleted immediately after use.
        </div>
    </div>

    <div class="tabs">
        <div class="tab-btn active" onclick="switchTab('predict')">📊 Prediction</div>
        <div class="tab-btn" onclick="switchTab('train')">🎓 Admin Training</div>
    </div>

    <div id="tab-predict" class="tab-content active">
        <span class="section-label">1. Select Report Type</span>
        <div class="report-grid">
            <div class="report-card active" onclick="selectReport('daily')">
                <input type="radio" name="reportType" value="daily" checked>
                <span class="card-title">Daily Report</span>
                <span class="card-desc">Defect Classification Only</span>
                <span class="check-icon k-icon k-i-check"></span>
            </div>
            
            <div class="report-card" onclick="selectReport('monthly')">
                <input type="radio" name="reportType" value="monthly">
                <span class="card-title">Monthly Report</span>
                <span class="card-desc">Category + Defect Class</span>
                <span class="check-icon k-icon k-i-check"></span>
            </div>
        </div>

        <span class="section-label">2. AI Configuration</span>
        <div class="config-box">
            <div>
                <div style="font-weight: 600; font-size:14px;">Deep Process Cleansing</div>
                <div style="font-size:12px; color: var(--text-sub);">Remove noise from data [Recommended ON].</div>
            </div>
            <label class="switch">
                <input type="checkbox" id="cleanPredict" checked>
                <span class="slider"></span>
            </label>
        </div>

        <span class="section-label">3. Upload File</span>
        <input name="filePredict" id="filePredict" type="file" />

        <button id="btnPredict" class="btn-process" disabled>
            <span class="k-icon k-i-gears"></span> RUN PREDICTION
        </button>
    </div>

    <div id="tab-train" class="tab-content">
        <div style="background:#FFFBEB; border:1px solid #FEF3C7; color:#92400E; padding:12px; border-radius:8px; margin-bottom:20px; font-size:13px;">
            <b>Admin Area:</b> Upload master data to retrain the AI models. This process may take a few minutes.
        </div>

        <span class="section-label">1. Training Configuration</span>
        <div class="config-box">
            <div>
                <div style="font-weight: 600; font-size:14px;">Deep Training Cleansing</div>
                <div style="font-size:12px; color: var(--text-sub);">Remove noise from data [Recommended ON].</div>
            </div>
            <label class="switch">
                <input type="checkbox" id="cleanTrain" checked>
                <span class="slider"></span>
            </label>
        </div>

        <span class="section-label">2. Upload Master Data</span>
        <input name="fileTrain" id="fileTrain" type="file" />

        <button id="btnTrain" class="btn-process btn-train" disabled>
            <span class="k-icon k-i-upload"></span> START RETRAINING
        </button>
    </div>

</div>

<div class="modal-overlay" id="loadingModal">
    <div class="loading-box">
        <div class="spinner" id="modalSpinner"></div>
        
        <div class="modal-icon" id="iconSuccess">✅</div>
        <div class="modal-icon" id="iconError">❌</div>

        <div class="loading-text" id="modalTitle">Processing...</div>
        <div class="loading-sub" id="modalSub">Please wait, AI is working.</div>

        <button class="btn-close-modal" id="btnCloseModal" onclick="closeModal()">Close</button>
    </div>
</div>

<script>
    // --- TABS LOGIC ---
    function switchTab(tab) {
        $(".tab-btn").removeClass("active");
        $(".tab-content").removeClass("active");
        
        if(tab === 'predict') {
            $(".tab-btn:eq(0)").addClass("active");
            $("#tab-predict").addClass("active");
        } else {
            $(".tab-btn:eq(1)").addClass("active");
            $("#tab-train").addClass("active");
        }
    }

    // --- SELECTION LOGIC ---
    function selectReport(type) {
        $(".report-card").removeClass("active");
        $("input[name='reportType'][value='" + type + "']").closest(".report-card").addClass("active");
        $("input[name='reportType'][value='" + type + "']").prop("checked", true);
    }

    function closeModal() {
        $("#loadingModal").fadeOut();
    }

    $(document).ready(function() {
        // --- KENDO UPLOAD INIT ---
        $("#filePredict").kendoUpload({
            multiple: false, validation: { allowedExtensions: [".xlsx", ".xls"] },
            select: function() { $("#btnPredict").removeAttr("disabled").css("opacity", "1"); },
            clear: function() { $("#btnPredict").attr("disabled", "disabled").css("opacity", "0.6"); }
        });

        $("#fileTrain").kendoUpload({
            multiple: false, validation: { allowedExtensions: [".xlsx", ".xls"] },
            select: function() { $("#btnTrain").removeAttr("disabled").css("opacity", "1"); },
            clear: function() { $("#btnTrain").attr("disabled", "disabled").css("opacity", "0.6"); }
        });

        // --- PREDICTION LOGIC ---
        $("#btnPredict").click(function() {
            var files = $("#filePredict").data("kendoUpload").getFiles();
            if (files.length === 0) return;

            var reportType = $("input[name='reportType']:checked").val();
            var isClean = $("#cleanPredict").is(":checked") ? "true" : "false";
            var formData = new FormData();
            formData.append("file", files[0].rawFile);

            // Show Loading with RAM Status
            $("#loadingModal").css("display", "flex").hide().fadeIn();
            $("#modalSpinner").show(); $(".modal-icon, #btnCloseModal").hide();
            $("#modalTitle").text("Analyzing Data...");
            $("#modalSub").text("Securely Processing in RAM..."); // <--- UPDATED INFO

            fetch("/predict?report_type=" + reportType + "&enable_cleansing=" + isClean, { method: "POST", body: formData })
            .then(res => { if(!res.ok) throw new Error("Server Error"); return res.blob(); })
            .then(blob => {
                $("#modalSpinner").hide(); $("#iconSuccess").fadeIn();
                $("#modalTitle").text("Success!"); 
                // <--- UPDATED INFO: Confirming Deletion
                $("#modalSub").text("Report Downloaded. Data wiped from memory."); 
                
                var url = window.URL.createObjectURL(blob);
                var a = document.createElement("a"); a.href = url;
                a.download = (reportType === 'daily' ? "DAILY_" : "MONTHLY_") + "REPORT_" + files[0].name;
                document.body.appendChild(a); a.click(); a.remove();
                
                setTimeout(closeModal, 3000);
            })
            .catch(err => {
                $("#modalSpinner").hide(); $("#iconError").fadeIn();
                $("#modalTitle").text("Failed"); $("#modalSub").text(err.message);
                $("#btnCloseModal").show();
            });
        });

        // --- TRAINING LOGIC (Background Polling) ---
        $("#btnTrain").click(function() {
            var files = $("#fileTrain").data("kendoUpload").getFiles();
            if (files.length === 0) return;

            var isClean = $("#cleanTrain").is(":checked") ? "true" : "false";
            var formData = new FormData();
            formData.append("file", files[0].rawFile);

            $("#loadingModal").css("display", "flex").hide().fadeIn();
            $("#modalSpinner").show(); $(".modal-icon, #btnCloseModal").hide();
            $("#modalTitle").text("Retraining AI...");
            $("#modalSub").text("Initializing...");

            // 1. Trigger Start
            fetch("/train?enable_cleansing=" + isClean, { method: "POST", body: formData })
            .then(response => {
                if(response.ok) {
                    // 2. Poll Status Invisibly
                    var poller = setInterval(function() {
                        fetch("/train/status")
                        .then(r => r.json())
                        .then(data => {
                            // Update Status Text Real-time
                            $("#modalSub").text(data.message); 

                            // Check Finish
                            if(data.progress >= 100) {
                                clearInterval(poller);
                                $("#modalSpinner").hide(); $("#iconSuccess").fadeIn();
                                $("#modalTitle").text("Training Complete!");
                                $("#modalSub").text("New models ready. Temporary files deleted.");
                                $("#btnCloseModal").show();
                            }
                            // Check Error
                            if(data.message && data.message.startsWith("Error")) {
                                clearInterval(poller);
                                $("#modalSpinner").hide(); $("#iconError").fadeIn();
                                $("#modalTitle").text("Training Failed");
                                $("#modalSub").text(data.message);
                                $("#btnCloseModal").show();
                            }
                        });
                    }, 1000);
                } else {
                    response.json().then(data => {
                       throw new Error(data.detail || "Failed to start training.");
                    });
                }
            })
            .catch(err => {
                $("#modalSpinner").hide(); $("#iconError").fadeIn();
                $("#modalTitle").text("Error"); $("#modalSub").text(err.message);
                $("#btnCloseModal").show();
            });
        });
    });
</script>

</body>
</html>