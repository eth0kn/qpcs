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
            --primary: #4F46E5;
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

        header { text-align: center; margin-bottom: 20px; }
        h1 { margin: 0; font-size: 24px; font-weight: 700; color: #111; letter-spacing: -0.5px; }
        .subtitle { color: var(--text-sub); font-size: 14px; margin-top: 5px; }

        .privacy-badge {
            background-color: #EFF6FF;
            border: 1px solid #DBEAFE;
            color: #1E40AF;
            padding: 12px; border-radius: 8px; font-size: 13px;
            display: flex; align-items: center; justify-content: center; gap: 8px;
            margin-bottom: 25px;
        }

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

        /* BUTTONS */
        .btn-process {
            width: 100%; background-color: var(--primary); color: white; border: none;
            padding: 16px; border-radius: 12px; font-size: 15px; font-weight: 600;
            cursor: pointer; transition: background 0.3s; margin-top: 20px;
            display: flex; align-items: center; justify-content: center; gap: 8px;
        }
        .btn-process:hover { background-color: var(--primary-hover); }
        .btn-process:disabled { background-color: #E5E7EB; color: #9CA3AF; cursor: not-allowed; }
        .btn-train { background-color: #111827; } 
        .btn-train:hover { background-color: #000000; }

        /* KENDO OVERRIDES */
        .k-upload { border-radius: 12px !important; border-color: var(--border) !important; }
        .k-dropzone { background: #fff !important; padding: 15px !important; }

        /* MODAL LOADING */
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

        <span class="section-label">2. Upload File</span>
        <input name="filePredict" id="filePredict" type="file" />

        <button id="btnPredict" class="btn-process" disabled>
            <span class="k-icon k-i-gears"></span> RUN PREDICTION
        </button>
    </div>

    <div id="tab-train" class="tab-content">
        <div style="background:#FFFBEB; border:1px solid #FEF3C7; color:#92400E; padding:12px; border-radius:8px; margin-bottom:20px; font-size:13px;">
            <b>Admin Area:</b> Upload master data to retrain the AI models. This process may take a few minutes.
        </div>

        <span class="section-label">1. Upload Master Data</span>
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
        <div class="loading-sub" id="modalSub">Initial Connection...</div>

        <button class="btn-close-modal" id="btnCloseModal" onclick="closeModal()">Close</button>
    </div>
</div>

<script>
    const API_BASE_URL = "http://13.229.172.201:8000"; 

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

    // --- GLOBAL POLLING ENGINE ---
    let pollingInterval = null;

    function startPolling(contextName) {
        if(pollingInterval) clearInterval(pollingInterval);
        
        pollingInterval = setInterval(function() {
            $.ajax({
                url: API_BASE_URL + "/progress?nocache=" + Math.random(),
                type: "GET",
                cache: false,
                success: function(data) {
                    console.log("Server Poll:", data);
                    
                    if (data && data.message) {
                         // Hanya update jika pesannya bukan "Idle" (default state)
                         if (data.message !== "Idle") {
                            $("#modalTitle").text(contextName + ": " + data.progress + "%");
                            $("#modalSub").text(data.message);
                         }
                    }

                    // Auto Stop Logic untuk Training
                    if(contextName === "Training") {
                        if (data.progress === 100 && !data.is_running) {
                            clearInterval(pollingInterval);
                            showResult("success", "Training Complete!", "New AI models ready.");
                        }
                        if (data.progress === 0 && !data.is_running && data.message.includes("Error")) {
                            clearInterval(pollingInterval);
                            showResult("error", "Training Failed", data.message);
                        }
                    }
                },
                error: function(err) {
                    console.warn("Polling Network Error:", err);
                }
            });
        }, 500);
    }

    function showResult(type, title, sub) {
        $("#modalSpinner").hide();
        $("#modalTitle").text(title);
        $("#modalSub").text(sub);
        $("#btnCloseModal").show();
        
        if(type === 'success') $("#iconSuccess").show();
        else $("#iconError").show();
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
            var formData = new FormData();
            formData.append("file", files[0].rawFile);

            // 1. Show Modal & Reset UI
            $("#loadingModal").css("display", "flex").hide().fadeIn();
            $("#modalSpinner").show(); $(".modal-icon, #btnCloseModal").hide();
            $("#modalTitle").text("Connecting...");
            $("#modalSub").text("Uploading file to server...");

            // 2. Start Polling IMMEDIATELY
            startPolling("Processing");

            // 3. Fetch Request (URL CLEANED: No cleansing params)
            fetch(API_BASE_URL + "/predict?report_type=" + reportType, { 
                method: "POST", 
                body: formData 
            })
            .then(res => { 
                if(!res.ok) return res.json().then(e => { throw new Error(e.detail || "Server Error") });
                return res.blob(); 
            })
            .then(blob => {
                if(pollingInterval) clearInterval(pollingInterval);
                showResult("success", "Success!", "Report Downloaded.");
                
                var url = window.URL.createObjectURL(blob);
                var a = document.createElement("a"); a.href = url;
                a.download = "RESULT_" + files[0].name;
                document.body.appendChild(a); a.click(); a.remove();
            })
            .catch(err => {
                if(pollingInterval) clearInterval(pollingInterval);
                showResult("error", "Failed", err.message);
            });
        });

        // --- TRAINING LOGIC ---
        $("#btnTrain").click(function() {
            var files = $("#fileTrain").data("kendoUpload").getFiles();
            if (files.length === 0) return;

            if(!confirm("Start Retraining? This will take 5-10 minutes.")) return;

            var formData = new FormData();
            formData.append("file", files[0].rawFile);

            // 1. Show Modal
            $("#loadingModal").css("display", "flex").hide().fadeIn();
            $("#modalSpinner").show(); $(".modal-icon, #btnCloseModal").hide();
            $("#modalTitle").text("Retraining AI...");
            $("#modalSub").text("Initializing upload...");

            // 2. Trigger Start (URL CLEANED: No cleansing params)
            fetch(API_BASE_URL + "/train", { method: "POST", body: formData })
            .then(res => res.json())
            .then(data => {
                startPolling("Training");
            })
            .catch(err => {
                showResult("error", "Connection Failed", "Could not reach server.");
            });
        });
    });
</script>

</body>
</html>