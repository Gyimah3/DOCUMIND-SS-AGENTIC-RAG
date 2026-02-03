const AUTH_STORAGE_KEY = 'documind_access_token';
const REFRESH_STORAGE_KEY = 'documind_refresh_token';
const USER_EMAIL_KEY = 'documind_user_email';
const SELECTED_INDEX_KEY = 'documind_selected_index';

document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const authView = document.getElementById('authView');
    const appView = document.getElementById('appView');
    const authError = document.getElementById('authError');
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    const showRegister = document.getElementById('showRegister');
    const showLogin = document.getElementById('showLogin');
    const logoutBtn = document.getElementById('logoutBtn');

    const userEmailText = document.getElementById('userEmailText');
    const userNameText = document.getElementById('userNameText');
    const avatarLetter = document.getElementById('avatarLetter');

    const chatInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const chatDisplay = document.getElementById('chatDisplay');
    const activeIndexName = document.getElementById('activeIndexName');

    const indexNameLabel = document.getElementById('indexNameLabel');

    const plusBtn = document.getElementById('plusBtn');
    const plusMenu = document.getElementById('plusMenu');
    const menuUpload = document.getElementById('menuUpload');
    const indexListMenu = document.getElementById('indexListMenu');

    const uploadModal = document.getElementById('uploadModal');
    const closeUploadModal = document.getElementById('closeUploadModal');
    const cancelUpload = document.getElementById('cancelUpload');
    const startUpload = document.getElementById('startUpload');
    const modalFileInput = document.getElementById('modalFileInput');
    const modalIndexName = document.getElementById('modalIndexName');

    const checkMultimodal = document.getElementById('checkMultimodal');
    const checkExtractTables = document.getElementById('checkExtractTables');
    const checkExtractImages = document.getElementById('checkExtractImages');
    const activeTagContainer = document.getElementById('activeTagContainer');
    const toastContainer = document.getElementById('toastContainer');

    const kbList = document.getElementById('kbList');

    const relevantChunks = document.getElementById('relevantChunks');
    const contextContent = document.getElementById('contextContent');

    const previewModal = document.getElementById('previewModal');
    const closePreviewModal = document.getElementById('closePreviewModal');
    const previewFrame = document.getElementById('previewFrame');
    const previewTitle = document.getElementById('previewTitle');
    const openExternalBtn = document.getElementById('openExternalBtn');

    const refreshAllDocsBtn = document.getElementById('refreshAllDocs');
    const allDocsTableBody = document.getElementById('allDocsTableBody');

    // State
    let currentThreadId = Math.random().toString(36).substring(7);
    let selectedIndexName = sessionStorage.getItem(SELECTED_INDEX_KEY) || null;

    // --- Authentication ---
    function showError(msg) {
        authError.textContent = msg;
        authError.classList.remove('hidden');
    }

    function clearError() {
        authError.textContent = '';
        authError.classList.add('hidden');
    }

    function showApp() {
        authView.classList.add('hidden');
        appView.classList.remove('hidden');
        const email = localStorage.getItem(USER_EMAIL_KEY) || 'User';
        userEmailText.textContent = email;
        userNameText.textContent = email.split('@')[0];
        avatarLetter.textContent = email[0].toUpperCase();

        updateIndexUI();
        loadIndexes();
    }

    function showAuth() {
        appView.classList.add('hidden');
        authView.classList.remove('hidden');
        localStorage.removeItem(AUTH_STORAGE_KEY);
        localStorage.removeItem(REFRESH_STORAGE_KEY);
        localStorage.removeItem(USER_EMAIL_KEY);
        clearError();
    }

    function getAuthHeaders() {
        const token = localStorage.getItem(AUTH_STORAGE_KEY);
        const headers = { 'Content-Type': 'application/json' };
        if (token) headers['Authorization'] = `Bearer ${token}`;
        return headers;
    }

    async function checkAuth() {
        const token = localStorage.getItem(AUTH_STORAGE_KEY);
        if (!token) return showAuth();
        try {
            const r = await fetch('/auth/verify-token', { headers: { Authorization: `Bearer ${token}` } });
            if (r.ok) {
                const data = await r.json();
                if (data.valid) return showApp();
            }
        } catch (_) { }
        showAuth();
    }

    const navItems = {
        chat: { btn: document.getElementById('navChat'), view: document.getElementById('chatView'), panel: true },
        docs: { btn: document.getElementById('navDocuments'), view: document.getElementById('documentsView'), panel: false },
        analytics: { btn: document.getElementById('navAnalytics'), view: document.getElementById('analyticsView'), panel: false },
        settings: { btn: document.getElementById('navSettings'), view: document.getElementById('settingsView'), panel: false }
    };

    function switchView(viewKey) {
        Object.keys(navItems).forEach(key => {
            const item = navItems[key];
            if (key === viewKey) {
                item.btn.classList.add('active');
                item.view.classList.remove('hidden');
                const sidePanel = document.querySelector('.side-panel');
                if (sidePanel) {
                    if (item.panel) sidePanel.classList.remove('hidden');
                    else sidePanel.classList.add('hidden');
                }
            } else {
                item.btn.classList.remove('active');
                item.view.classList.add('hidden');
            }
        });

        if (viewKey === 'docs') loadAllDocuments();
    }

    function showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        const icon = type === 'success' ?
            `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>` :
            `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;

        toast.innerHTML = `
            ${icon}
            <span>${message}</span>
        `;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('hiding');
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // --- Document Preview ---
    function openPreview(title, url) {
        if (!url) {
            showToast('Preview not available for this document', 'error');
            return;
        }

        previewTitle.textContent = title;
        if (openExternalBtn) openExternalBtn.href = url;

        // Check for Office documents
        const lowerUrl = url.toLowerCase();
        const lowerTitle = title.toLowerCase();

        if (lowerUrl.match(/\.(docx|doc|xlsx|xls|pptx|ppt|pdf)$/) || lowerTitle.match(/\.(docx|doc|xlsx|xls|pptx|ppt|pdf)$/)) {
            // Use Google Docs Viewer for Office files and PDFs (better compatibility)
            previewFrame.src = `https://docs.google.com/gview?url=${encodeURIComponent(url)}&embedded=true`;
        } else {
            // Native rendering for Images, Text
            previewFrame.src = url;
        }

        previewModal.classList.remove('hidden');
        // Small delay to allow display:flex to apply before adding opacity
        requestAnimationFrame(() => {
            previewModal.classList.add('show');
        });
    }

    function closePreview() {
        previewModal.classList.remove('show');
        setTimeout(() => {
            previewModal.classList.add('hidden');
            previewFrame.src = '';
        }, 300); // Wait for transition
    }

    closePreviewModal.onclick = closePreview;
    previewModal.onclick = (e) => {
        if (e.target === previewModal) closePreview();
    };

    navItems.chat.btn.onclick = () => switchView('chat');
    navItems.docs.btn.onclick = () => switchView('docs');
    navItems.analytics.btn.onclick = () => switchView('analytics');
    navItems.settings.btn.onclick = () => switchView('settings');

    // --- Index Management ---
    async function updateIndexUI() {
        const displayLabel = selectedIndexName ? selectedIndexName : 'All Documents';
        if (indexNameLabel) indexNameLabel.textContent = displayLabel;

        // Update input placeholder
        chatInput.placeholder = selectedIndexName ? `Message ${selectedIndexName}...` : 'Message all documents...';

        // Update tag chip visibility
        activeTagContainer.innerHTML = '';

        const chip = document.createElement('div');
        chip.className = 'tag-chip';

        if (selectedIndexName) {
            // Fetch URL immediately for the tag chip link
            let docUrl = null;
            try {
                const r = await fetch(`/api/v1/documents/${selectedIndexName}/url`, { headers: getAuthHeaders() });
                if (r.ok) {
                    const data = await r.json();
                    docUrl = data.url;
                }
            } catch (err) {
                console.warn('Could not fetch document URL for tag chip');
            }

            chip.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
                </svg>
                <span>${selectedIndexName}</span>
                <a href="${docUrl || '#'}" target="${docUrl ? '_blank' : '_self'}" class="tag-chip-view" title="View Document" style="display: flex; align-items: center; justify-content: center; margin-left: 6px; cursor: pointer; opacity: 0.8; transition: opacity 0.2s; color: inherit; text-decoration: none;">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                        <circle cx="12" cy="12" r="3"></circle>
                    </svg>
                </a>
                <div class="tag-chip-remove" title="Clear tag">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="12" height="12">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </div>
            `;
            chip.querySelector('.tag-chip-remove').onclick = (e) => {
                e.stopPropagation();
                setSelectedIndex(null);
                showToast('Switched to All Documents', 'info');
            };

            // If we have a URL, try to open preview first; if specific key pressed or mobile, link takes over
            chip.querySelector('.tag-chip-view').onclick = (e) => {
                e.stopPropagation();
                if (docUrl) {
                    e.preventDefault(); // Stop the link from opening immediately
                    openPreview(selectedIndexName, docUrl); // Open internal preview
                } else {
                    showToast('Document preview not available', 'error');
                }
            };

            activeTagContainer.appendChild(chip);
        } else {
            // ... (All Documents logic remains same or can be simplified)
            chip.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                </svg>
                <span>All Documents</span>
            `;
            activeTagContainer.appendChild(chip);
        }
        loadDocuments(); // Load docs for this index in KB panel
    }

    function setSelectedIndex(name) {
        selectedIndexName = name || null;
        if (name) sessionStorage.setItem(SELECTED_INDEX_KEY, name);
        else sessionStorage.removeItem(SELECTED_INDEX_KEY);
        updateIndexUI();
    }

    async function loadIndexes() {
        indexListMenu.innerHTML = '<div style="padding: 8px 12px; font-size: 12px; color: var(--text-muted);">Loading…</div>';
        try {
            const r = await fetch('/vectorstore/list-indexes', { headers: getAuthHeaders() });
            if (!r.ok) throw new Error();
            const list = await r.json();
            indexListMenu.innerHTML = '';

            // "All" option
            const allItem = document.createElement('div');
            allItem.className = 'index-item-mini' + (!selectedIndexName ? ' active' : '');
            allItem.textContent = 'All Documents';
            allItem.onclick = () => { setSelectedIndex(null); plusMenu.classList.remove('show'); };
            indexListMenu.appendChild(allItem);

            list.forEach(item => {
                const name = item.index_name;
                const div = document.createElement('div');
                div.className = 'index-item-mini' + (selectedIndexName === name ? ' active' : '');
                div.textContent = name;
                div.onclick = () => { setSelectedIndex(name); plusMenu.classList.remove('show'); };
                indexListMenu.appendChild(div);
            });
        } catch (err) {
            indexListMenu.innerHTML = '<div style="padding: 8px 12px; font-size: 12px; color: #ef4444;">Failed to load</div>';
        }
    }

    plusBtn.onclick = (e) => {
        e.stopPropagation();
        plusMenu.classList.toggle('show');
        if (plusMenu.classList.contains('show')) loadIndexes();
    };

    menuUpload.onclick = () => {
        plusMenu.classList.remove('show');
        uploadModal.classList.add('show');
        if (selectedIndexName) modalIndexName.value = selectedIndexName;
    };


    document.addEventListener('click', (e) => {
        if (!plusMenu.contains(e.target) && !plusBtn.contains(e.target)) {
            plusMenu.classList.remove('show');
        }
    });

    // Modal Events
    closeUploadModal.onclick = cancelUpload.onclick = () => {
        uploadModal.classList.remove('show');
        modalFileInput.value = '';
    };

    checkMultimodal.onchange = () => {
        if (checkMultimodal.checked) {
            checkExtractImages.checked = true;
            checkExtractImages.disabled = true;
        } else {
            checkExtractImages.disabled = false;
        }
    };

    startUpload.onclick = async () => {
        const file = modalFileInput.files[0];
        const idxName = modalIndexName.value.trim();
        if (!file || !idxName) return showToast('File and Document Identifier are required', 'error');

        const formData = new FormData();
        formData.append('files', file); // API expects 'files' list
        formData.append('index_name', idxName);
        formData.append('multimodal', checkMultimodal.checked);
        formData.append('extract_tables', checkExtractTables.checked);
        formData.append('extract_images', checkExtractImages.checked);

        startUpload.disabled = true;
        cancelUpload.disabled = true;
        closeUploadModal.disabled = true;

        // Progressive status messages
        const stages = [
            { text: 'Uploading file', icon: '\u2B06' },
            { text: 'Analyzing document', icon: '\uD83D\uDD0D' },
            { text: 'Extracting content', icon: '\u2699\uFE0F' },
            { text: 'Building index', icon: '\uD83D\uDCCA' },
        ];
        let stageIndex = 0;
        const dots = ['.', '..', '...'];
        let dotIndex = 0;

        function updateButtonText() {
            const stage = stages[stageIndex];
            startUpload.innerHTML = `<span class="upload-spinner"></span> ${stage.text}${dots[dotIndex]}`;
            dotIndex = (dotIndex + 1) % dots.length;
        }
        updateButtonText();

        // Cycle through dots every 500ms
        const dotInterval = setInterval(updateButtonText, 500);
        // Advance stage every 3s (upload is usually the longest)
        const stageInterval = setInterval(() => {
            if (stageIndex < stages.length - 1) {
                stageIndex++;
                dotIndex = 0;
                updateButtonText();
            }
        }, 3000);

        try {
            const r = await fetch('/vectorstore/add-document', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${localStorage.getItem(AUTH_STORAGE_KEY)}` },
                body: formData
            });

            clearInterval(dotInterval);
            clearInterval(stageInterval);

            if (r.ok) {
                startUpload.innerHTML = '<span style="color: #22c55e;">\u2713</span> Done!';
                await new Promise(res => setTimeout(res, 800));
                uploadModal.classList.remove('show');
                modalFileInput.value = '';
                showToast(`Successfully uploaded "${file.name}"`, 'success');
                // Refresh indexes and switch
                setSelectedIndex(idxName);
                loadIndexes();
            } else {
                const err = await r.json();
                showToast(err.detail || 'Upload failed', 'error');
            }
        } catch (err) {
            clearInterval(dotInterval);
            clearInterval(stageInterval);
            showToast('Upload failed due to network error', 'error');
        } finally {
            startUpload.disabled = false;
            cancelUpload.disabled = false;
            closeUploadModal.disabled = false;
            startUpload.textContent = 'Upload';
        }
    };

    // --- Panel Tabs Logic ---
    const panelViews = {
        docs: { btn: document.getElementById('tabDocs'), view: document.getElementById('docsViewPanel') },
        chunks: { btn: document.getElementById('tabChunks'), view: document.getElementById('chunksViewPanel') },
        context: { btn: document.getElementById('tabContext'), view: document.getElementById('contextViewPanel') }
    };

    function switchPanel(activeKey) {
        Object.keys(panelViews).forEach(key => {
            const item = panelViews[key];
            if (key === activeKey) {
                item.btn.classList.add('active');
                item.view.classList.remove('hidden');
            } else {
                item.btn.classList.remove('active');
                item.view.classList.add('hidden');
            }
        });
    }

    panelViews.docs.btn.onclick = () => switchPanel('docs');
    panelViews.chunks.btn.onclick = () => switchPanel('chunks');
    panelViews.context.btn.onclick = () => switchPanel('context');

    // --- Knowledge Base (Documents) ---

    // --- Global Documents Management ---
    async function loadAllDocuments() {
        if (!allDocsTableBody) return;
        allDocsTableBody.innerHTML = '<tr><td colspan="4" style="padding: 40px; text-align: center; color: var(--text-muted);"><div class="loading-spinner" style="margin: 0 auto 12px; border-width: 2px; width: 24px; height: 24px;"></div>Loading documents...</td></tr>';

        try {
            const r = await fetch('/api/v1/documents/list-all', { headers: getAuthHeaders() });
            if (!r.ok) throw new Error('Failed to load');
            const docs = await r.json();
            renderAllDocsTable(docs);
        } catch (err) {
            allDocsTableBody.innerHTML = '<tr><td colspan="4" style="padding: 40px; text-align: center; color: #ef4444;">Error loading documents. Please try again.</td></tr>';
        }
    }

    function renderAllDocsTable(docs) {
        allDocsTableBody.innerHTML = '';
        if (docs.length === 0) {
            allDocsTableBody.innerHTML = '<tr><td colspan="4" style="padding: 40px; text-align: center; color: var(--text-muted);">No documents found. Upload some to get started!</td></tr>';
            return;
        }

        docs.forEach(doc => {
            const date = new Date(doc.created_at).toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric'
            });
            const tr = document.createElement('tr');
            tr.className = 'doc-row';
            tr.innerHTML = `
                <td style="padding: 16px; font-weight: 500;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18" style="color: var(--primary-color);">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                        </svg>
                        <a href="${doc.s3_url || doc.url}" target="_blank" class="doc-title-link" style="color: inherit; text-decoration: none; font-weight: 500;" title="Open in new tab">
                            <span>${doc.title}</span>
                        </a>
                    </div>
                </td>
                <td style="padding: 16px;">
                    <span class="tag-chip" style="font-size: 11px; padding: 4px 10px; background: rgba(37, 99, 235, 0.08); border: 1px solid rgba(37, 99, 235, 0.1); color: var(--primary-color);">
                        ${doc.index_name}
                    </span>
                </td>
                <td style="padding: 16px; color: var(--text-muted);">${date}</td>
                <td style="padding: 16px; text-align: right;">
                    <div style="display: flex; gap: 8px; justify-content: flex-end;">
                        <button class="action-btn-view" title="View Document">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
                                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                                <circle cx="12" cy="12" r="3"></circle>
                            </svg>
                        </button>
                        <button class="action-btn-delete" title="Delete Document">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
                                <polyline points="3 6 5 6 21 6"></polyline>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                            </svg>
                        </button>
                    </div>
                </td>
            `;

            // Attach click handler to the title link to intercept default navigation
            const link = tr.querySelector('.doc-title-link');
            if (link) {
                link.onclick = (e) => {
                    e.preventDefault();
                    openPreview(doc.title, doc.s3_url || doc.url);
                };
            }

            tr.querySelector('.action-btn-view').onclick = () => openPreview(doc.title, doc.s3_url || doc.url);
            tr.querySelector('.action-btn-delete').onclick = () => confirmDeleteDocument(doc.id, doc.title);

            allDocsTableBody.appendChild(tr);
        });
    }

    async function confirmDeleteDocument(id, title) {
        if (!confirm(`Are you sure you want to permanently delete "${title}"?\n\nThis will remove the file from S3 and all its contents from the vector index. This action cannot be undone.`)) return;

        try {
            showToast('Deleting document...', 'info');
            const r = await fetch(`/api/v1/documents/${id}`, {
                method: 'DELETE',
                headers: getAuthHeaders()
            });
            if (r.ok || r.status === 204) {
                showToast('Document deleted successfully', 'success');
                loadAllDocuments();
                loadIndexes(); // Refresh KB list in case an index was cleared
            } else {
                throw new Error('Failed to delete');
            }
        } catch (err) {
            showToast('Error deleting document', 'error');
        }
    }

    if (refreshAllDocsBtn) refreshAllDocsBtn.onclick = loadAllDocuments;

    async function loadDocuments() {
        if (!selectedIndexName) {
            kbList.innerHTML = '<p style="padding: 20px; color: var(--text-muted); font-size: 13px;">Select an index to manage documents.</p>';
            return;
        }

        kbList.innerHTML = '<p style="padding: 20px; color: var(--text-muted); animation: pulse 1.5s infinite;">Loading documents...</p>';

        try {
            const r = await fetch(`/api/v1/documents/${selectedIndexName}/list`, { headers: getAuthHeaders() });
            const docs = await r.json();
            renderDocs(docs);
        } catch (err) {
            kbList.innerHTML = '<p style="padding: 20px; color: #ef4444;">Error loading documents.</p>';
        }
    }

    function renderDocs(docs) {
        kbList.innerHTML = '';
        if (docs.length === 0) {
            kbList.innerHTML = '<p style="padding: 20px; color: var(--text-muted); font-size: 13px;">No documents yet. Drag files or click upload icon to add.</p>';
            return;
        }

        docs.forEach(doc => {
            const item = document.createElement('div');
            item.className = 'kb-item';
            const date = new Date(doc.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

            item.innerHTML = `
                    <div class="kb-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                        </svg>
                    </div>
                    <div class="kb-details">
                        <a href="${doc.s3_url || doc.url}" target="_blank" class="kb-title" title="${doc.title}" style="text-decoration: none; color: inherit;">
                            ${doc.title}
                        </a>
                        <span class="kb-meta">${date}</span>
                    </div>
                `;

            // Attach click handler to link
            const link = item.querySelector('a.kb-title');
            if (link) {
                link.onclick = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    openPreview(doc.title, doc.s3_url || doc.url);
                };
            }

            item.onclick = () => openPreview(doc.title, doc.s3_url || doc.url);
            kbList.appendChild(item);
        });
    }

    // --- Upload Logic ---
    // Upload is now handled via the modal (menuUpload / startUpload)

    async function uploadFile(file) {
        if (!selectedIndexName) return showToast('Please select a document context first', 'error');

        const formData = new FormData();
        formData.append('file', file);

        const originalText = activeIndexName.textContent;
        activeIndexName.textContent = `Uploading ${file.name}...`;

        try {
            const r = await fetch(`/documents/${selectedIndexName}/upload`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${localStorage.getItem(AUTH_STORAGE_KEY)}` },
                body: formData
            });

            if (r.ok) {
                loadDocuments();
            } else {
                const err = await r.json();
                alert(`Upload failed: ${err.detail}`);
            }
        } catch (err) {
            alert('Upload failed due to network error');
        } finally {
            activeIndexName.textContent = originalText;
        }
    }

    // --- Chat Logic ---
    function appendMessage(content, isUser, usedDocs = []) {
        const msgDiv = document.createElement('div');
        msgDiv.className = (isUser ? 'user-message' : 'ai-message') + ' message';

        let html = marked.parse(content);

        if (!isUser && usedDocs.length > 0) {
            let sourcesHtml = '<div class="ai-message-footer">';
            sourcesHtml += '<span style="font-size: 11px; color: var(--text-muted); font-weight: 600;">SOURCES:</span>';
            const uniqueFiles = [...new Set(usedDocs.map(d => d.metadata?.filename).filter(Boolean))];
            uniqueFiles.forEach(file => {
                sourcesHtml += `<span class="source-tag">${file}</span>`;
            });
            sourcesHtml += '</div>';
            html += sourcesHtml;
        }

        msgDiv.innerHTML = html;
        chatDisplay.appendChild(msgDiv);
        chatDisplay.scrollTop = chatDisplay.scrollHeight;
        return msgDiv;
    }

    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        // Clear previous context/chunks state
        relevantChunks.innerHTML = '<p style="padding: 20px; color: var(--text-muted); font-size: 13px;">No chunks retrieved for this message.</p>';
        contextContent.innerHTML = '<p style="padding: 20px; color: var(--text-muted); font-size: 13px;">Select a chunk to view details.</p>';

        appendMessage(message, true);
        chatInput.value = '';

        const aiMsgDiv = appendMessage('.', false);
        let dotCount = 1;
        const typingInterval = setInterval(() => {
            dotCount = (dotCount % 3) + 1;
            aiMsgDiv.textContent = '.'.repeat(dotCount);
        }, 400);
        let accumulatedText = '';

        try {
            const params = new URLSearchParams({ top_k: 5 });
            if (selectedIndexName) params.append('index_name', selectedIndexName);

            const url = `/chat/invoke/${currentThreadId}?${params.toString()}`;
            const response = await fetch(url, {
                method: 'POST',
                headers: getAuthHeaders(),
                body: JSON.stringify({ message })
            });

            if (!response.ok) throw new Error();

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let usedDocs = [];

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const data = JSON.parse(line);
                        if (data.stream) {
                            clearInterval(typingInterval);
                            accumulatedText += data.stream;
                            aiMsgDiv.innerHTML = marked.parse(accumulatedText);
                        } else if (data.used_docs) {
                            usedDocs = data.used_docs;
                            updateContext(usedDocs);
                        }
                    } catch (e) { }
                }
            }

            // Re-render with sources
            aiMsgDiv.innerHTML = marked.parse(accumulatedText);
            if (usedDocs.length > 0) {
                const footer = document.createElement('div');
                footer.className = 'ai-message-footer';
                footer.style.borderTop = '2px solid var(--primary-light)';
                footer.style.marginTop = '16px';
                footer.style.paddingTop = '12px';
                footer.innerHTML = '<span style="font-size: 11px; color: var(--text-muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Source File:</span>';

                // Group pages/rows by filename
                const fileGroups = {};
                usedDocs.forEach(d => {
                    const fname = d.metadata?.filename || 'Unknown';
                    if (!fileGroups[fname]) fileGroups[fname] = { pages: new Set(), sheets: {} };
                    const sheetName = d.metadata?.sheet_name;
                    const rowNumber = d.metadata?.row_number;
                    const page = d.metadata?.page_number;
                    if (sheetName) {
                        if (!fileGroups[fname].sheets[sheetName]) fileGroups[fname].sheets[sheetName] = new Set();
                        if (rowNumber) fileGroups[fname].sheets[sheetName].add(rowNumber);
                    } else if (page !== undefined && page !== null) {
                        fileGroups[fname].pages.add(page);
                    }
                });

                Object.keys(fileGroups).forEach(fname => {
                    const group = fileGroups[fname];
                    const sheetNames = Object.keys(group.sheets);
                    let pageStr = '';
                    if (sheetNames.length > 0) {
                        const parts = sheetNames.map(s => {
                            const rows = Array.from(group.sheets[s]).sort((a, b) => a - b);
                            return rows.length > 0 ? `Sheet "${s}", Row ${rows.join(', ')}` : `Sheet "${s}"`;
                        });
                        pageStr = ` (${parts.join('; ')})`;
                    } else {
                        const pages = Array.from(group.pages).sort((a, b) => a - b);
                        pageStr = pages.length > 0 ? ` (Page ${pages.join(', ')})` : '';
                    }

                    footer.innerHTML += `<div style="display: flex; align-items: center; gap: 8px; margin-top: 8px;">
                        <div style="width: 24px; height: 24px; background: #fff1f2; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #e11d48;">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
                        </div>
                        <span style="font-size: 13px; font-weight: 600; color: var(--text-main);">${fname}${pageStr}</span>
                    </div>`;
                });
                aiMsgDiv.appendChild(footer);

                // Automatically switch to chunks view when answer arrives
                switchPanel('chunks');
            }

        } catch (err) {
            clearInterval(typingInterval);
            aiMsgDiv.textContent = 'Sorry, I encountered an error.';
        }
        chatDisplay.scrollTop = chatDisplay.scrollHeight;
    }

    function updateContext(docs) {
        relevantChunks.innerHTML = '<h3 style="font-size: 14px; margin-bottom: 12px; color: var(--text-muted);">Relevant Chunks</h3>';
        contextContent.innerHTML = '<h3 style="font-size: 14px; margin-bottom: 12px; color: var(--text-muted);">Content Preview</h3>';

        docs.forEach((doc, i) => {
            const chunkNumber = i + 1;
            const item = document.createElement('div');
            item.className = 'kb-item';
            item.style.marginBottom = '4px';
            item.innerHTML = `
                <div class="kb-icon" style="width: 32px; height: 32px; font-size: 12px; font-weight: 700;">${chunkNumber}</div>
                <div class="kb-details"><span class="kb-title">${doc.metadata?.filename || 'Chunk'}</span></div>
            `;
            item.onclick = () => {
                // Switch to context tab
                switchPanel('context');

                const locationInfo = doc.metadata?.sheet_name
                    ? `<strong>Sheet:</strong> ${doc.metadata.sheet_name}<br><strong>Row:</strong> ${doc.metadata.row_number || 'N/A'}`
                    : `<strong>Page:</strong> ${doc.metadata?.page_number || 'N/A'}`;
                contextContent.innerHTML = `
                    <h3 style="font-size: 14px; margin-bottom: 12px; color: var(--text-muted);">Chunk #${chunkNumber} Details</h3>
                    <div style="font-size: 13px; color: var(--text-muted); margin-bottom: 10px;">
                        <strong>File:</strong> ${doc.metadata?.filename || 'Unknown'}<br>
                        ${locationInfo}<br>
                        <strong>Score:</strong> ${doc.metadata?.vector_score ? Number(doc.metadata.vector_score).toFixed(4) : 'N/A'}
                    </div>
                    <div style="font-size: 14px; line-height: 1.6; color: var(--text-main); background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;">
                        ${doc.page_content}
                    </div>
                `;
            };
            relevantChunks.appendChild(item);
        });

        // Set initial preview if context is empty
        if (docs.length > 0 && contextContent.innerHTML.includes('Content Preview')) {
            // We don't call onclick here to avoid switching tab immediately on load
            // but we can populate the content
            const doc = docs[0];
            const initLocationInfo = doc.metadata?.sheet_name
                ? `<strong>Sheet:</strong> ${doc.metadata.sheet_name}<br><strong>Row:</strong> ${doc.metadata.row_number || 'N/A'}`
                : `<strong>Page:</strong> ${doc.metadata?.page_number || 'N/A'}`;
            contextContent.innerHTML = `
                <h3 style="font-size: 14px; margin-bottom: 12px; color: var(--text-muted);">Chunk #1 Details</h3>
                <div style="font-size: 13px; color: var(--text-muted); margin-bottom: 10px;">
                    <strong>File:</strong> ${doc.metadata?.filename || 'Unknown'}<br>
                    ${initLocationInfo}<br>
                    <strong>Score:</strong> ${doc.metadata?.vector_score ? Number(doc.metadata.vector_score).toFixed(4) : 'N/A'}
                </div>
                <div style="font-size: 14px; line-height: 1.6; color: var(--text-main); background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    ${doc.page_content}
                </div>
            `;
        }
    }

    // --- Event Listeners ---
    sendButton.onclick = sendMessage;
    chatInput.onkeypress = (e) => { if (e.key === 'Enter') sendMessage(); };

    loginForm.onsubmit = async (e) => {
        e.preventDefault();
        const email = document.getElementById('loginEmail').value;
        const password = document.getElementById('loginPassword').value;
        const res = await fetch('/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });
        if (res.ok) {
            const data = await res.json();
            localStorage.setItem(AUTH_STORAGE_KEY, data.access_token);
            localStorage.setItem(USER_EMAIL_KEY, email);
            showApp();
        } else {
            showError('Invalid email or password');
        }
    };

    registerForm.onsubmit = async (e) => {
        e.preventDefault();
        const payload = {
            first_name: document.getElementById('regFirst').value,
            last_name: document.getElementById('regLast').value,
            email: document.getElementById('regEmail').value,
            password: document.getElementById('regPassword').value,
        };
        const res = await fetch('/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (res.ok) {
            showError('Account created! Please sign in.');
            document.getElementById('showLogin').click();
        } else {
            showError('Registration failed');
        }
    };

    showRegister.onclick = () => { loginForm.classList.add('hidden'); registerForm.classList.remove('hidden'); clearError(); };
    showLogin.onclick = () => { registerForm.classList.add('hidden'); loginForm.classList.remove('hidden'); clearError(); };

    logoutBtn.onclick = () => showAuth();

    checkAuth();
});
