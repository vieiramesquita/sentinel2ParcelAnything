/**
 * LAPIG HQ-SAM Paddock Segmenter - Client Application Logic
 * Padrão Oficial: Sentinel-2 (Segmentação por Amostragem com 1 Clique = 1 Piquete Individual)
 * Auxiliares Visuais: Google Satellite e ESRI World Imagery
 */

document.addEventListener('DOMContentLoaded', () => {
    if (window.lucide) {
        lucide.createIcons();
    }

    // =========================================================================
    // Estado da Aplicação
    // =========================================================================
    const state = {
        currentMode: 'positive',   // 'positive' (incluir pasto) ou 'negative' (excluir/recortar)
        activeRoi: null,           // Dados do ROI atual retornado pelo backend
        savedPaddocks: [],         // [{ id, name, area_ha, confidence_score, geojson, layer, points }]
        lastPaddockId: null,       // ID do último piquete criado
        overlayLayer: null,        // L.ImageOverlay da imagem Sentinel-2 do imóvel
        carLayer: null,            // L.GeoJSON do limite do imóvel (CAR)
        paddocksLayer: null,       // L.FeatureGroup com todos os piquetes individuais salvos
        activeBasemap: 's2',       // 's2', 'google', 'esri'
        isOverlayVisible: true,    // Exibição da sobreposição Sentinel-2

        // Sessão Ativa de Rascunho / Refinamento Multi-Pontos
        candidatePoints: [],       // [{ id, lon, lat, label, marker }]
        candidateMask: null,       // { mask_index, confidence_score, area_ha, geojson }
        candidateLayer: null,      // L.GeoJSON com o preview do polígono em refinamento
        editingPaddockId: null     // ID do piquete que está sendo editado/refinado (null se for novo)
    };

    // =========================================================================
    // Elementos do DOM
    // =========================================================================
    const dom = {
        // Tabs
        tabRoiBtn: document.getElementById('tab-roi-btn'),
        tabPaddocksBtn: document.getElementById('tab-paddocks-btn'),
        tabRoiContent: document.getElementById('tab-roi-content'),
        tabPaddocksContent: document.getElementById('tab-paddocks-content'),
        tabPaddocksCount: document.getElementById('tab-paddocks-count'),

        // Header Metrics
        headerCount: document.getElementById('header-paddocks-count'),
        headerArea: document.getElementById('header-paddocks-area'),
        btnExportGeojson: document.getElementById('btn-export-geojson'),

        // ROI Form Elements
        presetSelector: document.getElementById('preset-selector'),
        btnLoadPreset: document.getElementById('btn-load-preset'),
        inputLat: document.getElementById('input-lat'),
        inputLon: document.getElementById('input-lon'),
        inputBuffer: document.getElementById('input-buffer'),
        btnLoadCoords: document.getElementById('btn-load-coords'),
        inputCarId: document.getElementById('input-car-id'),
        btnLoadCar: document.getElementById('btn-load-car'),
        fileDropzone: document.getElementById('file-dropzone'),
        fileInput: document.getElementById('file-input'),

        // Map HUDs
        activeRoiBanner: document.getElementById('active-roi-banner'),
        activeRoiLabel: document.getElementById('active-roi-label'),
        hudCoords: document.getElementById('hud-coords'),
        mapElement: document.getElementById('map'),

        // Basemap Switcher Buttons
        btnBmS2: document.getElementById('btn-bm-s2'),
        btnBmGoogle: document.getElementById('btn-bm-google'),
        btnBmEsri: document.getElementById('btn-bm-esri'),

        // Sampling Dock Controls
        btnModePositive: document.getElementById('btn-mode-positive'),
        btnModeNegative: document.getElementById('btn-mode-negative'),
        btnConfirmPaddock: document.getElementById('btn-confirm-paddock'),
        confirmBtnLabel: document.getElementById('confirm-btn-label'),
        btnUndoPoint: document.getElementById('btn-undo-point'),
        btnClearPoints: document.getElementById('btn-clear-points'),
        btnToggleOverlay: document.getElementById('btn-toggle-overlay'),
        dockTitleText: document.getElementById('dock-title-text'),
        dockStatusHint: document.getElementById('dock-status-hint'),
        candidatePointsBadge: document.getElementById('candidate-points-badge'),
        candidateScoreBadge: document.getElementById('candidate-score-badge'),
        candidateAreaBadge: document.getElementById('candidate-area-badge'),
        opacitySliderRow: document.getElementById('opacity-slider-row'),
        opacitySlider: document.getElementById('overlay-opacity-slider'),
        opacityVal: document.getElementById('opacity-val'),

        // Paddocks Management
        paddocksStatCount: document.getElementById('paddocks-stat-count'),
        paddocksStatArea: document.getElementById('paddocks-stat-area'),
        paddocksStatAvg: document.getElementById('paddocks-stat-avg'),
        paddocksList: document.getElementById('paddocks-list'),
        btnClearAllPaddocks: document.getElementById('btn-clear-all-paddocks'),

        // Loading Overlay & Progress Bar
        loadingOverlay: document.getElementById('loading-overlay'),
        loadingTitle: document.getElementById('loading-title'),
        loadingDesc: document.getElementById('loading-desc'),
        loadingBarFill: document.getElementById('loading-bar-fill'),
        loadingSubdetail: document.getElementById('loading-subdetail'),
        loadingPercentText: document.getElementById('loading-percent-text')
    };

    // =========================================================================
    // Inicialização do Leaflet Map e Basemaps
    // =========================================================================
    // 1. Sentinel-2 Cloudless (Padrão)
    const s2Basemap = L.tileLayer('https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/g/{z}/{y}/{x}.jpg', {
        maxZoom: 19,
        maxNativeZoom: 14,
        attribution: 'Sentinel-2 Cloudless &copy; EOX IT Services'
    });

    // 2. Google Satellite (Auxiliar de Alta Resolução)
    const googleSat = L.tileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
        maxZoom: 21,
        maxNativeZoom: 20,
        attribution: 'Google Satellite'
    });

    // 3. ESRI World Imagery (Auxiliar de Alta Resolução)
    const esriWorld = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 21,
        maxNativeZoom: 18,
        attribution: 'Esri World Imagery'
    });

    const map = L.map('map', {
        center: [-18.3720, -50.0050],
        zoom: 14,
        layers: [s2Basemap],
        zoomControl: false
    });

    L.control.zoom({ position: 'topleft' }).addTo(map);

    // FeatureGroup para armazenar todos os piquetes individuais
    state.paddocksLayer = L.featureGroup().addTo(map);

    // =========================================================================
    // Alternância de Basemaps Visuais
    // =========================================================================
    function setBasemap(type) {
        state.activeBasemap = type;
        map.removeLayer(s2Basemap);
        map.removeLayer(googleSat);
        map.removeLayer(esriWorld);

        dom.btnBmS2.classList.remove('active');
        dom.btnBmGoogle.classList.remove('active');
        dom.btnBmEsri.classList.remove('active');

        if (type === 's2') {
            s2Basemap.addTo(map);
            s2Basemap.bringToBack();
            dom.btnBmS2.classList.add('active');
        } else if (type === 'google') {
            googleSat.addTo(map);
            googleSat.bringToBack();
            dom.btnBmGoogle.classList.add('active');
        } else if (type === 'esri') {
            esriWorld.addTo(map);
            esriWorld.bringToBack();
            dom.btnBmEsri.classList.add('active');
        }
    }

    dom.btnBmS2.addEventListener('click', () => setBasemap('s2'));
    dom.btnBmGoogle.addEventListener('click', () => setBasemap('google'));
    dom.btnBmEsri.addEventListener('click', () => setBasemap('esri'));

    // Alternar visibilidade da sobreposição do Sentinel-2
    dom.btnToggleOverlay.addEventListener('click', () => {
        if (!state.overlayLayer) return;
        state.isOverlayVisible = !state.isOverlayVisible;

        if (state.isOverlayVisible) {
            state.overlayLayer.addTo(map);
            state.overlayLayer.setOpacity(parseFloat(dom.opacitySlider.value) / 100.0);
            dom.btnToggleOverlay.innerHTML = '<i data-lucide="layers"></i> <span>Sentinel: Ligado</span>';
            dom.btnToggleOverlay.classList.remove('text-rose');
        } else {
            map.removeLayer(state.overlayLayer);
            dom.btnToggleOverlay.innerHTML = '<i data-lucide="layers"></i> <span>Sentinel: Oculto</span>';
            dom.btnToggleOverlay.classList.add('text-rose');
        }
        if (window.lucide) lucide.createIcons();
    });

    // Slider de Transparência do Sentinel-2
    dom.opacitySlider.addEventListener('input', (e) => {
        const val = e.target.value;
        dom.opacityVal.innerText = `${val}%`;
        if (state.overlayLayer && state.isOverlayVisible) {
            state.overlayLayer.setOpacity(parseFloat(val) / 100.0);
        }
    });

    // Coordenadas Live do Mouse
    map.on('mousemove', (e) => {
        dom.hudCoords.innerText = `Lat: ${e.latlng.lat.toFixed(5)} | Lon: ${e.latlng.lng.toFixed(5)}`;
    });

    // =========================================================================
    // Utilitários de Loading e Notificação com Barra de Progresso em Tempo Real
    // =========================================================================
    let progressInterval = null;

    function startProgressPolling(initialTitle, initialDesc) {
        if (progressInterval) clearInterval(progressInterval);

        dom.loadingTitle.innerText = initialTitle || "Obtendo Imagens Sentinel-2...";
        dom.loadingDesc.innerText = initialDesc || "Iniciando comunicação com serviço orbital...";
        dom.loadingBarFill.style.width = '8%';
        dom.loadingPercentText.innerText = '8%';
        dom.loadingSubdetail.innerText = 'Conectando ao serviço...';
        dom.loadingOverlay.style.display = 'flex';

        progressInterval = setInterval(async () => {
            try {
                const res = await fetch('/api/progress');
                if (res.ok) {
                    const prog = await res.json();
                    if (prog.percent !== undefined) {
                        dom.loadingBarFill.style.width = `${prog.percent}%`;
                        dom.loadingPercentText.innerText = `${prog.percent}%`;
                        if (prog.title) dom.loadingTitle.innerText = prog.title;
                        if (prog.message) dom.loadingDesc.innerText = prog.message;
                        if (prog.detail) dom.loadingSubdetail.innerText = prog.detail;
                    }
                }
            } catch (e) {
                // silencioso
            }
        }, 200);
    }

    function stopProgressPolling() {
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }
        dom.loadingBarFill.style.width = '100%';
        dom.loadingPercentText.innerText = '100%';
        dom.loadingTitle.innerText = "Concluído!";
        dom.loadingDesc.innerText = "Imagem Sentinel-2 carregada no HQ-SAM!";
        dom.loadingSubdetail.innerText = "Pronto para cliques";

        setTimeout(() => {
            dom.loadingOverlay.style.display = 'none';
        }, 350);
    }

    function showLoading(title, desc) {
        startProgressPolling(title, desc);
    }

    function hideLoading() {
        stopProgressPolling();
    }

    // =========================================================================
    // Troca de Tabs
    // =========================================================================
    dom.tabRoiBtn.addEventListener('click', () => {
        dom.tabRoiBtn.classList.add('active');
        dom.tabPaddocksBtn.classList.remove('active');
        dom.tabRoiContent.classList.add('active');
        dom.tabPaddocksContent.classList.remove('active');
    });

    dom.tabPaddocksBtn.addEventListener('click', () => {
        dom.tabPaddocksBtn.classList.add('active');
        dom.tabRoiBtn.classList.remove('active');
        dom.tabPaddocksContent.classList.add('active');
        dom.tabRoiContent.classList.remove('active');
    });

    // =========================================================================
    // Carregar Lista de Presets (Demonstração)
    // =========================================================================
    async function fetchPresets() {
        try {
            const res = await fetch('/api/presets');
            const presets = await res.json();
            dom.presetSelector.innerHTML = '';

            if (presets.length === 0) {
                dom.presetSelector.innerHTML = '<option value="">Nenhuma fazenda encontrada em results/</option>';
                return;
            }

            presets.forEach((p, idx) => {
                const opt = document.createElement('option');
                opt.value = p.id;
                opt.innerText = `${idx + 1}. ${p.title} (${p.id.slice(0, 16)}...)`;
                dom.presetSelector.appendChild(opt);
            });
        } catch (err) {
            console.error("Erro ao buscar presets:", err);
            dom.presetSelector.innerHTML = '<option value="">Erro ao carregar presets</option>';
        }
    }
    fetchPresets();

    // =========================================================================
    // Aplicação da ROI Carregada
    // =========================================================================
    function applyLoadedRoi(data) {
        state.activeRoi = data;
        dom.activeRoiLabel.innerHTML = `<strong>Área Ativa:</strong> ${data.roi_name || data.roi_id} <span class="tag-default">Sentinel-2</span>`;
        dom.mapElement.classList.add('sampling-mode');

        // Limpa estado anterior
        clearAllPaddocks();

        // Enquadra mapa nos bounds WGS84
        if (data.bounds_wgs84) {
            const b = data.bounds_wgs84; // [min_lon, min_lat, max_lon, max_lat]
            const bounds = L.latLngBounds([b[1], b[0]], [b[3], b[2]]);
            map.fitBounds(bounds, { padding: [40, 40], maxZoom: 17 });

            // Adiciona overlay raster Sentinel-2 do imóvel
            if (data.has_overlay) {
                if (state.overlayLayer) {
                    map.removeLayer(state.overlayLayer);
                }
                const cacheBuster = Date.now();
                state.overlayLayer = L.imageOverlay(`/api/overlay?t=${cacheBuster}`, bounds, {
                    opacity: parseFloat(dom.opacitySlider.value) / 100.0,
                    interactive: false
                }).addTo(map);
                state.isOverlayVisible = true;
                dom.btnToggleOverlay.innerHTML = '<i data-lucide="layers"></i> <span>Sentinel: Ligado</span>';
                dom.btnToggleOverlay.classList.remove('text-rose');
            }
        }

        // Desenha limite do CAR se disponível
        if (state.carLayer) {
            map.removeLayer(state.carLayer);
            state.carLayer = null;
        }

        if (data.car_geojson) {
            state.carLayer = L.geoJSON(data.car_geojson, {
                style: {
                    color: '#f59e0b',
                    weight: 2.5,
                    dashArray: '6, 6',
                    fillColor: '#f59e0b',
                    fillOpacity: 0.05
                }
            }).addTo(map);
        }

        if (window.lucide) lucide.createIcons();
    }

    // Carregar Preset
    dom.btnLoadPreset.addEventListener('click', async () => {
        const presetId = dom.presetSelector.value;
        if (!presetId) return;

        showLoading("Carregando Sentinel-2 no HQ-SAM...", "Preparando imagem Sentinel-2 NDVI multi-temporal na GPU RTX 2080 Ti...");
        try {
            const res = await fetch('/api/roi/preset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ preset_id: presetId })
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Falha ao carregar preset");
            }
            const data = await res.json();
            applyLoadedRoi(data);
        } catch (err) {
            alert(`Erro: ${err.message}`);
        } finally {
            hideLoading();
        }
    });

    // Carregar Coordenadas
    dom.btnLoadCoords.addEventListener('click', async () => {
        const lat = parseFloat(dom.inputLat.value);
        const lon = parseFloat(dom.inputLon.value);
        const buffer = parseFloat(dom.inputBuffer.value) || 2000;

        if (isNaN(lat) || isNaN(lon)) {
            alert("Por favor, informe latitude e longitude válidas.");
            return;
        }

        showLoading("Obtendo Sentinel-2 Oficial...", `Carregando mosaico Sentinel-2 para (${lat.toFixed(4)}, ${lon.toFixed(4)})...`);
        try {
            const res = await fetch('/api/roi/coords', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ lat, lon, buffer_m: buffer })
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Falha ao carregar coordenadas");
            }
            const data = await res.json();
            applyLoadedRoi(data);
        } catch (err) {
            alert(`Erro: ${err.message}`);
        } finally {
            hideLoading();
        }
    });

    // Carregar CAR ID
    dom.btnLoadCar.addEventListener('click', async () => {
        const carId = dom.inputCarId.value.trim();
        if (!carId) {
            alert("Informe o código CAR do imóvel.");
            return;
        }

        showLoading("Localizando CAR e Sentinel-2...", `Buscando imóvel ${carId}...`);
        try {
            const res = await fetch('/api/roi/car_id', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ car_id: carId })
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Código CAR não encontrado");
            }
            const data = await res.json();
            applyLoadedRoi(data);
        } catch (err) {
            alert(`Erro: ${err.message}`);
        } finally {
            hideLoading();
        }
    });

    // Upload de Arquivo Vetorial
    dom.fileDropzone.addEventListener('click', () => dom.fileInput.click());
    dom.fileDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dom.fileDropzone.classList.add('dragover');
    });
    dom.fileDropzone.addEventListener('dragleave', () => dom.fileDropzone.classList.remove('dragover'));
    dom.fileDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dom.fileDropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            uploadVectorFile(e.dataTransfer.files[0]);
        }
    });
    dom.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadVectorFile(e.target.files[0]);
        }
    });

    async function uploadVectorFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        showLoading("Enviando e Processando Vetor...", `Lendo limites de ${file.name} e preparando Sentinel-2 correspondente...`);
        try {
            const res = await fetch('/api/roi/upload', {
                method: 'POST',
                body: formData
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Falha no upload");
            }
            const data = await res.json();
            applyLoadedRoi(data);
        } catch (err) {
            alert(`Erro no upload: ${err.message}`);
        } finally {
            hideLoading();
        }
    }

    // =========================================================================
    // Modo de Amostragem (Ponto Positivo (+) vs Ponto Negativo (−))
    // =========================================================================
    dom.btnModePositive.addEventListener('click', () => {
        state.currentMode = 'positive';
        dom.btnModePositive.classList.add('active');
        dom.btnModeNegative.classList.remove('active');
    });

    dom.btnModeNegative.addEventListener('click', () => {
        state.currentMode = 'negative';
        dom.btnModeNegative.classList.add('active');
        dom.btnModePositive.classList.remove('active');
    });

    // =========================================================================
    // CLIQUE NO MAPA: ADIÇÃO DE PONTOS DE AMOSTRAGEM / REFINAMENTO
    // =========================================================================
    map.on('click', async (e) => {
        if (!state.activeRoi) {
            alert("Selecione primeiro uma fazenda ou coordenada na barra lateral.");
            return;
        }

        const lon = e.latlng.lng;
        const lat = e.latlng.lat;
        const label = state.currentMode === 'positive' ? 1 : 0;

        await addPromptPoint(lat, lon, label);
    });

    /**
     * Adiciona um ponto de amostragem (positivo ou negativo) com marcador interativo
     */
    async function addPromptPoint(lat, lon, label) {
        const pointId = Date.now() + Math.random();

        // Marcador interativo com classe positiva (verde) ou negativa (vermelho)
        const markerIcon = L.divIcon({
            className: `sam-prompt-icon ${label === 1 ? 'positive' : 'negative'}`,
            html: label === 1 ? '+' : '−',
            iconSize: [22, 22],
            iconAnchor: [11, 11]
        });

        const marker = L.marker([lat, lon], { icon: markerIcon }).addTo(map);
        marker.bindTooltip("Clique para remover este ponto", { direction: 'top', offset: [0, -10] });

        // Clique no próprio marcador remove o ponto imediatamente!
        marker.on('click', async (ev) => {
            L.DomEvent.stopPropagation(ev);
            await removePromptPoint(pointId);
        });

        state.candidatePoints.push({
            id: pointId,
            lon,
            lat,
            label,
            marker
        });

        await refreshCandidatePrediction();
    }

    /**
     * Remove um ponto específico pelo seu ID e recalcula a máscara
     */
    async function removePromptPoint(pointId) {
        const idx = state.candidatePoints.findIndex(p => p.id === pointId);
        if (idx !== -1) {
            const removed = state.candidatePoints.splice(idx, 1)[0];
            if (removed.marker) map.removeLayer(removed.marker);
        }
        await refreshCandidatePrediction();
    }

    /**
     * Executa a inferência HQ-SAM com a lista atual de pontos e renderiza o preview do polígono
     */
    async function refreshCandidatePrediction() {
        const posCount = state.candidatePoints.filter(p => p.label === 1).length;
        const negCount = state.candidatePoints.filter(p => p.label === 0).length;
        const total = state.candidatePoints.length;

        // Se todos os pontos foram removidos
        if (total === 0) {
            if (state.candidateLayer) {
                map.removeLayer(state.candidateLayer);
                state.candidateLayer = null;
            }
            state.candidateMask = null;
            dom.btnConfirmPaddock.disabled = true;
            dom.btnConfirmPaddock.classList.remove('ready');
            dom.btnUndoPoint.disabled = true;
            dom.btnClearPoints.disabled = true;

            dom.candidatePointsBadge.style.display = 'none';
            dom.candidateScoreBadge.style.display = 'none';
            dom.candidateAreaBadge.style.display = 'none';

            dom.dockTitleText.innerHTML = '<i data-lucide="crosshair"></i> Amostragem & Refinamento de Piquete';
            dom.dockStatusHint.innerText = 'Clique no mapa para adicionar pontos de amostragem no pasto';
            if (window.lucide) lucide.createIcons();
            return;
        }

        // Habilita botões de controle de pontos
        dom.btnUndoPoint.disabled = false;
        dom.btnClearPoints.disabled = false;

        // Atualiza badge de pontos
        dom.candidatePointsBadge.innerText = `${total} pt (${posCount}+ / ${negCount}−)`;
        dom.candidatePointsBadge.style.display = 'inline-block';

        // É necessário ao menos 1 ponto positivo para delimitar o piquete
        if (posCount === 0) {
            if (state.candidateLayer) {
                map.removeLayer(state.candidateLayer);
                state.candidateLayer = null;
            }
            state.candidateMask = null;
            dom.btnConfirmPaddock.disabled = true;
            dom.btnConfirmPaddock.classList.remove('ready');
            dom.candidateScoreBadge.style.display = 'none';
            dom.candidateAreaBadge.style.display = 'none';
            dom.dockStatusHint.innerText = 'Adicione ao menos um ponto positivo (+) no interior do pasto para segmentar.';
            return;
        }

        // Dispara predição no backend HQ-SAM
        dom.dockStatusHint.innerText = 'Calculando segmentação no Sentinel-2 com HQ-SAM (GPU)...';
        try {
            const payload = state.candidatePoints.map(p => ({ lon: p.lon, lat: p.lat, label: p.label }));
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ points: payload, multimask: false })
            });

            if (!res.ok) {
                const err = await res.json();
                console.error("Erro na predição:", err);
                return;
            }

            const data = await res.json();
            const bestMask = data.best_mask;

            if (bestMask) {
                state.candidateMask = bestMask;

                // Renderiza preview dinâmico com contorno ciano neon tracejado
                if (state.candidateLayer) {
                    map.removeLayer(state.candidateLayer);
                }

                state.candidateLayer = L.geoJSON(bestMask.geojson, {
                    style: {
                        color: '#06b6d4',          // Ciano neon
                        weight: 3,
                        dashArray: '6, 6',
                        fillColor: '#10b981',      // Esmeralda
                        fillOpacity: 0.38
                    }
                }).addTo(map);

                // Atualiza badges
                dom.candidateScoreBadge.innerText = `Confiança: ${bestMask.confidence_score}%`;
                dom.candidateScoreBadge.style.display = 'inline-block';
                dom.candidateAreaBadge.innerText = `${bestMask.area_ha} ha`;
                dom.candidateAreaBadge.style.display = 'inline-block';

                // Habilita botão de confirmação com animação de pulso
                dom.btnConfirmPaddock.disabled = false;
                dom.btnConfirmPaddock.classList.add('ready');

                if (state.editingPaddockId) {
                    dom.dockTitleText.innerHTML = '<i data-lucide="edit-3"></i> Refinando Piquete Existente';
                    dom.dockStatusHint.innerText = `Área calculada: ${bestMask.area_ha} ha. Adicione/remova pontos ou confirme as alterações.`;
                } else {
                    dom.dockTitleText.innerHTML = '<i data-lucide="crosshair"></i> Rascunho de Piquete Proposto';
                    dom.dockStatusHint.innerText = `Polígono gerado: ${bestMask.area_ha} ha (${bestMask.confidence_score}%). Confirme para salvar.`;
                }
            } else {
                dom.dockStatusHint.innerText = 'Nenhum piquete identificado nesta combinação de pontos. Ajuste os cliques.';
                dom.btnConfirmPaddock.disabled = true;
                dom.btnConfirmPaddock.classList.remove('ready');
            }
        } catch (err) {
            console.error("Falha ao atualizar predição:", err);
            dom.dockStatusHint.innerText = `Erro ao processar: ${err.message}`;
        }

        if (window.lucide) lucide.createIcons();
    }

    // =========================================================================
    // Desfazer Último Ponto Adicionado
    // =========================================================================
    dom.btnUndoPoint.addEventListener('click', async () => {
        if (state.candidatePoints.length === 0) return;
        const last = state.candidatePoints.pop();
        if (last && last.marker) map.removeLayer(last.marker);
        await refreshCandidatePrediction();
    });

    // =========================================================================
    // Cancelar e Limpar Amostras da Sessão Ativa
    // =========================================================================
    dom.btnClearPoints.addEventListener('click', () => {
        clearCandidateSession();
    });

    function clearCandidateSession() {
        state.candidatePoints.forEach(p => {
            if (p.marker) map.removeLayer(p.marker);
        });
        state.candidatePoints = [];

        if (state.candidateLayer) {
            map.removeLayer(state.candidateLayer);
            state.candidateLayer = null;
        }
        state.candidateMask = null;

        // Se estava editando um piquete salvo, restaura opacidade original de seu layer
        if (state.editingPaddockId) {
            const p = state.savedPaddocks.find(item => item.id === state.editingPaddockId);
            if (p && p.layer) {
                p.layer.setStyle({ fillOpacity: 0.40, opacity: 1.0 });
            }
            state.editingPaddockId = null;
        }

        dom.confirmBtnLabel.innerText = "Confirmar Piquete";
        dom.btnConfirmPaddock.disabled = true;
        dom.btnConfirmPaddock.classList.remove('ready');
        dom.btnUndoPoint.disabled = true;
        dom.btnClearPoints.disabled = true;

        dom.candidatePointsBadge.style.display = 'none';
        dom.candidateScoreBadge.style.display = 'none';
        dom.candidateAreaBadge.style.display = 'none';

        dom.dockTitleText.innerHTML = '<i data-lucide="crosshair"></i> Amostragem & Refinamento de Piquete';
        dom.dockStatusHint.innerText = 'Clique no mapa para adicionar pontos de amostragem no pasto';

        // Volta para o modo positivo
        state.currentMode = 'positive';
        dom.btnModePositive.classList.add('active');
        dom.btnModeNegative.classList.remove('active');

        if (window.lucide) lucide.createIcons();
    }

    // =========================================================================
    // CONFIRMAÇÃO DO POLÍGONO (SALVAMENTO DEFINITIVO)
    // =========================================================================
    dom.btnConfirmPaddock.addEventListener('click', async () => {
        if (!state.candidateMask) return;

        const pointsData = state.candidatePoints.map(p => ({ lon: p.lon, lat: p.lat, label: p.label }));

        if (state.editingPaddockId) {
            // =================================================================
            // ATUALIZAÇÃO DE PIQUETE EXISTENTE
            // =================================================================
            const pIdx = state.savedPaddocks.findIndex(p => p.id === state.editingPaddockId);
            if (pIdx !== -1) {
                const current = state.savedPaddocks[pIdx];
                try {
                    const res = await fetch(`/api/paddock/${current.id}`, {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            name: current.name,
                            area_ha: state.candidateMask.area_ha,
                            confidence_score: state.candidateMask.confidence_score,
                            geojson: state.candidateMask.geojson,
                            points: pointsData
                        })
                    });

                    if (!res.ok) {
                        const err = await res.json();
                        alert(`Erro ao atualizar: ${err.detail || 'Falha no backend'}`);
                        return;
                    }

                    // Remove camada anterior
                    if (current.layer) map.removeLayer(current.layer);

                    // Atualiza dados locais
                    current.area_ha = state.candidateMask.area_ha;
                    current.confidence_score = state.candidateMask.confidence_score;
                    current.geojson = state.candidateMask.geojson;
                    current.points = pointsData;

                    // Cria nova camada Leaflet definitiva
                    current.layer = createPaddockMapLayer(current);

                    clearCandidateSession();
                    updatePaddocksStats();
                    renderPaddocksList();
                } catch (err) {
                    alert(`Falha ao salvar alterações: ${err.message}`);
                }
            }
        } else {
            // =================================================================
            // CRIAÇÃO E CONFIRMAÇÃO DE NOVO PIQUETE
            // =================================================================
            const paddockNumber = state.savedPaddocks.length + 1;
            const paddockName = `Piquete ${paddockNumber}`;

            try {
                const res = await fetch('/api/paddock/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: paddockName,
                        area_ha: state.candidateMask.area_ha,
                        confidence_score: state.candidateMask.confidence_score,
                        geojson: state.candidateMask.geojson,
                        points: pointsData
                    })
                });

                if (!res.ok) {
                    const err = await res.json();
                    alert(`Erro ao salvar: ${err.detail || 'Falha no backend'}`);
                    return;
                }

                const data = await res.json();
                const record = data.paddock;

                // Cria camada Leaflet permanente
                const newPaddock = {
                    id: record.id,
                    name: record.name,
                    area_ha: record.area_ha,
                    confidence_score: record.confidence_score,
                    geojson: record.geojson,
                    points: pointsData
                };
                newPaddock.layer = createPaddockMapLayer(newPaddock);

                state.savedPaddocks.push(newPaddock);
                state.lastPaddockId = record.id;

                clearCandidateSession();
                updatePaddocksStats();
                renderPaddocksList();
            } catch (err) {
                alert(`Falha ao salvar piquete: ${err.message}`);
            }
        }
    });

    /**
     * Cria a camada visual permanente do piquete no Leaflet
     */
    function createPaddockMapLayer(p) {
        const layer = L.geoJSON(p.geojson, {
            style: {
                color: '#10b981',
                weight: 2.5,
                fillColor: '#10b981',
                fillOpacity: 0.40
            }
        }).bindPopup(`
            <div style="font-family: 'Plus Jakarta Sans', sans-serif;">
                <strong style="color: #10b981; font-size: 0.95rem;">${p.name}</strong><br>
                <span style="font-size: 0.8rem; color: #94a3b8;">Área: <strong>${p.area_ha} ha</strong></span><br>
                <span style="font-size: 0.8rem; color: #94a3b8;">Confiança HQ-SAM: <strong>${p.confidence_score}%</strong></span>
                <div style="margin-top: 10px;">
                    <button class="btn-primary" style="padding: 5px 10px; font-size: 0.74rem; width: 100%; justify-content: center;" onclick="window.editPaddockFromPopup('${p.id}')">
                        <i data-lucide="edit-3"></i> Refinar este Piquete
                    </button>
                </div>
            </div>
        `).addTo(state.paddocksLayer);

        layer.bindTooltip(`${p.name} (${p.area_ha} ha)`, {
            permanent: true,
            direction: 'center',
            className: 'glass-pill font-mono'
        });

        return layer;
    }

    /**
     * Inicia o modo de refinamento de um piquete já existente
     */
    function startEditPaddock(id) {
        const p = state.savedPaddocks.find(item => item.id === id);
        if (!p) return;

        // Limpa sessão ativa se houver
        clearCandidateSession();

        state.editingPaddockId = p.id;
        dom.confirmBtnLabel.innerText = "Atualizar Piquete";

        // Deixa a camada atual com transparência suave para visualização
        if (p.layer) {
            p.layer.setStyle({ fillOpacity: 0.15, opacity: 0.4 });
        }

        // Restaura pontos de amostragem
        if (p.points && p.points.length > 0) {
            p.points.forEach(pt => {
                const markerIcon = L.divIcon({
                    className: `sam-prompt-icon ${pt.label === 1 ? 'positive' : 'negative'}`,
                    html: pt.label === 1 ? '+' : '−',
                    iconSize: [22, 22],
                    iconAnchor: [11, 11]
                });
                const marker = L.marker([pt.lat, pt.lon], { icon: markerIcon }).addTo(map);
                marker.bindTooltip("Clique para remover este ponto", { direction: 'top', offset: [0, -10] });
                const pointId = Date.now() + Math.random();
                marker.on('click', async (ev) => {
                    L.DomEvent.stopPropagation(ev);
                    await removePromptPoint(pointId);
                });
                state.candidatePoints.push({
                    id: pointId,
                    lon: pt.lon,
                    lat: pt.lat,
                    label: pt.label,
                    marker
                });
            });
        }

        refreshCandidatePrediction();

        // Enquadra o piquete no mapa
        const tempLayer = L.geoJSON(p.geojson);
        map.fitBounds(tempLayer.getBounds(), { padding: [60, 60], maxZoom: 18 });
    }

    window.editPaddockFromPopup = (id) => {
        map.closePopup();
        startEditPaddock(id);
    };

    // =========================================================================
    // Gerenciamento e Exibição de Piquetes
    // =========================================================================
    function updatePaddocksStats() {
        const count = state.savedPaddocks.length;
        const totalArea = state.savedPaddocks.reduce((sum, p) => sum + p.area_ha, 0);
        const avgArea = count > 0 ? (totalArea / count) : 0;

        dom.headerCount.innerText = count;
        dom.headerArea.innerText = `${totalArea.toFixed(2)} ha`;
        dom.tabPaddocksCount.innerText = count;

        dom.paddocksStatCount.innerText = count;
        dom.paddocksStatArea.innerText = totalArea.toFixed(1);
        dom.paddocksStatAvg.innerText = avgArea.toFixed(1);

        dom.btnExportGeojson.disabled = count === 0;
    }

    function renderPaddocksList() {
        dom.paddocksList.innerHTML = '';

        if (state.savedPaddocks.length === 0) {
            dom.paddocksList.innerHTML = `
                <div class="empty-state">
                    <i data-lucide="mouse-pointer-click" class="empty-icon"></i>
                    <p>Nenhum piquete criado ainda.</p>
                    <span>Clique sobre o pasto no mapa. Adicione/remova pontos de refinamento e clique em 'Confirmar Piquete' para salvar!</span>
                </div>
            `;
            if (window.lucide) lucide.createIcons();
            return;
        }

        state.savedPaddocks.forEach((p, idx) => {
            const card = document.createElement('div');
            card.className = 'paddock-card';
            card.innerHTML = `
                <div class="paddock-info">
                    <span class="paddock-name">${p.name}</span>
                    <div class="paddock-meta">
                        <span class="paddock-meta-badge">${p.area_ha} ha</span>
                        <span>•</span>
                        <span>Confiança: ${p.confidence_score}%</span>
                    </div>
                </div>
                <div class="paddock-actions">
                    <button class="btn-edit-paddock" data-id="${p.id}" title="Refinar e ajustar polígono deste piquete">
                        <i data-lucide="edit-3"></i>
                    </button>
                    <button class="btn-del-paddock" data-id="${p.id}" title="Excluir piquete">
                        <i data-lucide="trash"></i>
                    </button>
                </div>
            `;

            // Clique no card foca no piquete
            card.addEventListener('click', (e) => {
                if (e.target.closest('.paddock-actions')) return;
                const tempLayer = L.geoJSON(p.geojson);
                map.fitBounds(tempLayer.getBounds(), { padding: [50, 50], maxZoom: 18 });
            });

            // Botão refinar piquete
            const btnEdit = card.querySelector('.btn-edit-paddock');
            btnEdit.addEventListener('click', (e) => {
                e.stopPropagation();
                startEditPaddock(p.id);
            });

            // Botão excluir piquete
            const btnDel = card.querySelector('.btn-del-paddock');
            btnDel.addEventListener('click', async (e) => {
                e.stopPropagation();
                await deletePaddock(p.id);
            });

            dom.paddocksList.appendChild(card);
        });

        if (window.lucide) lucide.createIcons();
    }

    async function deletePaddock(id) {
        try {
            await fetch(`/api/paddock/${id}`, { method: 'DELETE' });
            const pIdx = state.savedPaddocks.findIndex(p => p.id === id);
            if (pIdx !== -1) {
                const removed = state.savedPaddocks.splice(pIdx, 1)[0];
                if (removed.layer) map.removeLayer(removed.layer);
            }

            if (state.editingPaddockId === id) {
                clearCandidateSession();
            }

            state.lastPaddockId = state.savedPaddocks.length > 0 ? state.savedPaddocks[state.savedPaddocks.length - 1].id : null;

            renderPaddocksList();
            updatePaddocksStats();
        } catch (err) {
            console.error("Erro ao deletar piquete:", err);
        }
    }

    function clearAllPaddocks() {
        clearCandidateSession();
        state.savedPaddocks.forEach(p => {
            if (p.layer) map.removeLayer(p.layer);
        });
        state.savedPaddocks = [];
        state.lastPaddockId = null;
        state.paddocksLayer.clearLayers();
        renderPaddocksList();
        updatePaddocksStats();
    }

    dom.btnClearAllPaddocks.addEventListener('click', async () => {
        if (state.savedPaddocks.length === 0) return;
        if (!confirm("Deseja realmente excluir todos os piquetes da sessão?")) return;

        try {
            await fetch('/api/paddocks/clear', { method: 'POST' });
            clearAllPaddocks();
        } catch (err) {
            console.error("Erro ao limpar piquetes:", err);
        }
    });

    // =========================================================================
    // Exportação em GeoJSON
    // =========================================================================
    dom.btnExportGeojson.addEventListener('click', () => {
        if (state.savedPaddocks.length === 0) return;
        window.location.href = '/api/export/geojson';
    });
});
