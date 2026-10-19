/**
 * LAPIG HQ-SAM Paddock Segmenter - Client-Side Demonstration for GitHub Pages
 * Laboratório de Processamento de Imagens e Geoprocessamento (LAPIG / UFG)
 * 100% Static, 100% Client-Side via Leaflet + Turf.js
 */

document.addEventListener('DOMContentLoaded', () => {
    if (window.lucide) {
        lucide.createIcons();
    }

    // =========================================================================
    // Estado da Aplicação
    // =========================================================================
    const state = {
        presets: [],               // Lista carregada de data/manifest.json
        activePreset: null,        // Preset atualmente selecionado
        loadedGroundtruth: [],     // Polígonos de piquetes do modelo HQ-SAM da fazenda ativa
        savedPaddocks: [],         // Piquetes confirmados pelo usuário [{ id, name, area_ha, confidence_score, geojson, layer, points }]
        activeBasemap: 's2',       // 's2', 'google', 'esri'
        isOverlayVisible: true,    // Visibilidade da camada Sentinel-2

        // Layers Leaflet
        overlayLayer: null,
        carLayer: null,
        paddocksLayer: null,
        candidateLayer: null,

        // Sessão de Rascunho / Amostragem Ativa
        currentMode: 'positive',   // 'positive' (+) ou 'negative' (−)
        candidatePoints: [],       // [{ id, lon, lat, label, marker }]
        candidateMask: null,       // { area_ha, confidence_score, geojson }
        editingPaddockId: null     // ID do piquete em edição ou null se novo
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

        // Preset Selector & Farm Meta Card
        presetSelector: document.getElementById('preset-selector'),
        btnLoadPreset: document.getElementById('btn-load-preset'),
        metaMunicipio: document.getElementById('meta-farm-municipio'),
        metaArea: document.getElementById('meta-farm-area'),
        metaPaddocks: document.getElementById('meta-farm-paddocks'),
        metaCar: document.getElementById('meta-farm-car'),

        // Map HUDs
        activeRoiBanner: document.getElementById('active-roi-banner'),
        activeRoiLabel: document.getElementById('active-roi-label'),
        hudCoords: document.getElementById('hud-coords'),
        btnBmS2: document.getElementById('btn-bm-s2'),
        btnBmGoogle: document.getElementById('btn-bm-google'),
        btnBmEsri: document.getElementById('btn-bm-esri'),

        // Dock Amostragem & Refinamento
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
        opacitySlider: document.getElementById('overlay-opacity-slider'),
        opacityVal: document.getElementById('opacity-val'),

        // Paddock List & Stats
        paddocksStatCount: document.getElementById('paddocks-stat-count'),
        paddocksStatArea: document.getElementById('paddocks-stat-area'),
        paddocksStatAvg: document.getElementById('paddocks-stat-avg'),
        paddocksList: document.getElementById('paddocks-list'),
        btnClearAllPaddocks: document.getElementById('btn-clear-all-paddocks'),

        // Loading Overlay
        loadingOverlay: document.getElementById('loading-overlay'),
        loadingTitle: document.getElementById('loading-title'),
        loadingDesc: document.getElementById('loading-desc'),
        loadingBarFill: document.getElementById('loading-bar-fill')
    };

    // =========================================================================
    // Inicialização do Mapa Leaflet
    // =========================================================================
    const map = L.map('map', {
        center: [-18.3720, -50.0050],
        zoom: 13,
        zoomControl: false,
        attributionControl: false
    });

    L.control.zoom({ position: 'bottomright' }).addTo(map);
    setTimeout(() => map.invalidateSize(), 150);
    window.addEventListener('resize', () => map.invalidateSize());

    // Basemaps Tile Layers
    const baseLayers = {
        s2: L.tileLayer('https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/g/{z}/{y}/{x}.jpg', {
            maxZoom: 18,
            attribution: 'Sentinel-2 Cloudless • EOX'
        }),
        google: L.tileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
            maxZoom: 20,
            attribution: 'Google Satellite'
        }),
        esri: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            maxZoom: 19,
            attribution: 'Esri World Imagery'
        })
    };

    // Adiciona o basemap padrão Sentinel-2
    baseLayers.s2.addTo(map);

    // Grupo de camadas de piquetes salvos
    state.paddocksLayer = L.featureGroup().addTo(map);

    // Atualiza HUD de coordenadas e zoom
    map.on('mousemove', (e) => {
        dom.hudCoords.innerText = `Lat: ${e.latlng.lat.toFixed(5)} | Lon: ${e.latlng.lng.toFixed(5)} | Zoom: ${map.getZoom()}`;
    });

    // =========================================================================
    // Alternância de Basemaps
    // =========================================================================
    function switchBasemap(type) {
        state.activeBasemap = type;
        [dom.btnBmS2, dom.btnBmGoogle, dom.btnBmEsri].forEach(b => b.classList.remove('active'));

        if (type === 's2') {
            dom.btnBmS2.classList.add('active');
            map.removeLayer(baseLayers.google);
            map.removeLayer(baseLayers.esri);
            baseLayers.s2.addTo(map);
        } else if (type === 'google') {
            dom.btnBmGoogle.classList.add('active');
            map.removeLayer(baseLayers.s2);
            map.removeLayer(baseLayers.esri);
            baseLayers.google.addTo(map);
        } else if (type === 'esri') {
            dom.btnBmEsri.classList.add('active');
            map.removeLayer(baseLayers.s2);
            map.removeLayer(baseLayers.google);
            baseLayers.esri.addTo(map);
        }
    }

    dom.btnBmS2.addEventListener('click', () => switchBasemap('s2'));
    dom.btnBmGoogle.addEventListener('click', () => switchBasemap('google'));
    dom.btnBmEsri.addEventListener('click', () => switchBasemap('esri'));

    // Slider de Opacidade
    dom.opacitySlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value, 10);
        dom.opacityVal.innerText = `${val}%`;
        if (state.overlayLayer) {
            state.overlayLayer.setOpacity(val / 100);
        }
    });

    // Botão de Visibilidade do Overlay
    dom.btnToggleOverlay.addEventListener('click', () => {
        state.isOverlayVisible = !state.isOverlayVisible;
        if (state.overlayLayer) {
            if (state.isOverlayVisible) {
                state.overlayLayer.setOpacity(parseInt(dom.opacitySlider.value, 10) / 100);
                dom.btnToggleOverlay.innerHTML = '<i data-lucide="eye"></i>';
            } else {
                state.overlayLayer.setOpacity(0);
                dom.btnToggleOverlay.innerHTML = '<i data-lucide="eye-off"></i>';
            }
            if (window.lucide) lucide.createIcons();
        }
    });

    // Alternador de Abas
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
    // Carregamento de Presets e Metadados (data/manifest.json)
    // =========================================================================
    async function initPresets() {
        try {
            const res = await fetch('data/manifest.json');
            if (!res.ok) throw new Error("Manifest não encontrado");
            const presets = await res.json();
            state.presets = presets;

            dom.presetSelector.innerHTML = '';
            presets.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p.id;
                opt.innerText = `${p.title} (${p.car_area_ha.toLocaleString('pt-BR', {minimumFractionDigits: 1})} ha)`;
                dom.presetSelector.appendChild(opt);
            });

            // Carrega automaticamente o primeiro preset
            if (presets.length > 0) {
                loadPreset(presets[0].id);
            }
        } catch (err) {
            console.error("Falha ao carregar presets:", err);
            dom.presetSelector.innerHTML = '<option value="">Erro ao carregar dados</option>';
        }
    }

    dom.btnLoadPreset.addEventListener('click', () => {
        const id = dom.presetSelector.value;
        if (id) loadPreset(id);
    });

    dom.presetSelector.addEventListener('change', () => {
        const id = dom.presetSelector.value;
        if (id) loadPreset(id);
    });

    /**
     * Carrega uma fazenda de demonstração
     */
    async function loadPreset(presetId) {
        const preset = state.presets.find(p => p.id === presetId);
        if (!preset) return;

        state.activePreset = preset;
        clearCandidateSession();

        showLoading("Carregando Imóvel Demonstrativo...", `Preparando limites de ${preset.title} e mosaico Sentinel-2...`);
        setLoadingProgress(25);

        try {
            // Atualiza Card de Metadados
            dom.metaMunicipio.innerText = preset.municipio;
            dom.metaArea.innerText = `${preset.car_area_ha.toLocaleString('pt-BR', {minimumFractionDigits: 1})} ha`;
            dom.metaPaddocks.innerText = `${preset.paddocks_count} piquetes catalogados`;
            dom.metaCar.innerText = preset.id;

            // Atualiza banner superior
            dom.activeRoiBanner.style.display = 'block';
            dom.activeRoiLabel.innerText = `${preset.title} • ${preset.car_area_ha.toLocaleString('pt-BR', {maximumFractionDigits: 0})} ha`;

            setLoadingProgress(50);

            // 1. Carrega limite CAR
            if (state.carLayer) {
                map.removeLayer(state.carLayer);
                state.carLayer = null;
            }
            try {
                const carRes = await fetch(preset.car_url);
                if (carRes.ok) {
                    const carJson = await carRes.json();
                    if (carJson) {
                        state.carLayer = L.geoJSON(carJson, {
                            style: {
                                color: '#f59e0b',
                                weight: 2.5,
                                dashArray: '4, 4',
                                fillColor: '#f59e0b',
                                fillOpacity: 0.04
                            }
                        }).addTo(map);
                    }
                }
            } catch (e) {
                console.warn("CAR geometry fetch skipped:", e);
            }

            setLoadingProgress(75);

            // 2. Carrega Overlay PNG do Sentinel-2
            if (state.overlayLayer) {
                map.removeLayer(state.overlayLayer);
                state.overlayLayer = null;
            }
            const opacity = parseInt(dom.opacitySlider.value, 10) / 100;
            state.overlayLayer = L.imageOverlay(preset.overlay_url, preset.overlay_bounds, {
                opacity: opacity,
                interactive: false
            }).addTo(map);

            // 3. Carrega Piquetes Groundtruth do modelo HQ-SAM
            const padRes = await fetch(preset.paddocks_url);
            if (!padRes.ok) throw new Error("Falha ao ler polígonos dos piquetes");
            const padData = await padRes.json();
            state.loadedGroundtruth = padData.features || [];

            setLoadingProgress(100);

            // Enquadra a fazenda no mapa
            map.invalidateSize();
            map.fitBounds(preset.bounds, { padding: [40, 40], maxZoom: 16 });

            dom.dockStatusHint.innerText = `Pronto! Clique em qualquer pasto para amostragem interativa.`;
        } catch (err) {
            alert(`Erro ao carregar dados do imóvel: ${err.message}`);
        } finally {
            setTimeout(hideLoading, 300);
        }
    }

    // =========================================================================
    // Modo de Amostragem (+ Incluir / − Excluir)
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
    // Clique no Mapa: Amostragem e Refinamento
    // =========================================================================
    map.on('click', (e) => {
        if (!state.activePreset || state.loadedGroundtruth.length === 0) {
            alert("Aguarde o carregamento do imóvel demonstrativo.");
            return;
        }

        const lon = e.latlng.lng;
        const lat = e.latlng.lat;
        const label = state.currentMode === 'positive' ? 1 : 0;

        addPromptPoint(lat, lon, label);
    });

    /**
     * Adiciona marcador no mapa e recalcula o polígono
     */
    function addPromptPoint(lat, lon, label) {
        const pointId = Date.now() + Math.random();

        const markerIcon = L.divIcon({
            className: `sam-prompt-icon ${label === 1 ? 'positive' : 'negative'}`,
            html: label === 1 ? '+' : '−',
            iconSize: [22, 22],
            iconAnchor: [11, 11]
        });

        const marker = L.marker([lat, lon], { icon: markerIcon }).addTo(map);
        marker.bindTooltip("Clique para remover este ponto", { direction: 'top', offset: [0, -10] });

        // Clique no próprio marcador remove o ponto imediatamente
        marker.on('click', (ev) => {
            L.DomEvent.stopPropagation(ev);
            removePromptPoint(pointId);
        });

        state.candidatePoints.push({
            id: pointId,
            lon,
            lat,
            label,
            marker
        });

        refreshCandidatePrediction();
    }

    function removePromptPoint(pointId) {
        const idx = state.candidatePoints.findIndex(p => p.id === pointId);
        if (idx !== -1) {
            const removed = state.candidatePoints.splice(idx, 1)[0];
            if (removed.marker) map.removeLayer(removed.marker);
        }
        refreshCandidatePrediction();
    }

    // =========================================================================
    // Motor de Inferência Espacial Client-Side (Turf.js)
    // =========================================================================
    function refreshCandidatePrediction() {
        const posPoints = state.candidatePoints.filter(p => p.label === 1);
        const negPoints = state.candidatePoints.filter(p => p.label === 0);
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

        dom.btnUndoPoint.disabled = false;
        dom.btnClearPoints.disabled = false;
        dom.candidatePointsBadge.innerText = `${total} pt (${posPoints.length}+ / ${negPoints.length}−)`;
        dom.candidatePointsBadge.style.display = 'inline-block';

        // É necessário ao menos 1 ponto positivo (+) para formar o piquete
        if (posPoints.length === 0) {
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

        // 1. Identifica polígonos que contêm os pontos positivos
        let matchedPolygons = [];
        posPoints.forEach(pt => {
            const turfPt = turf.point([pt.lon, pt.lat]);
            let found = null;

            // Busca por contenção direta
            for (let i = 0; i < state.loadedGroundtruth.length; i++) {
                const feat = state.loadedGroundtruth[i];
                if (turf.booleanPointInPolygon(turfPt, feat)) {
                    found = feat;
                    break;
                }
            }

            // Fallback: se o clique foi próximo à cerca, pega o polígono mais próximo (< 120m)
            if (!found) {
                let minDistance = Infinity;
                let closest = null;
                for (let i = 0; i < state.loadedGroundtruth.length; i++) {
                    const feat = state.loadedGroundtruth[i];
                    try {
                        const dist = turf.pointToPolygonDistance(turfPt, feat, { units: 'kilometers' }) * 1000.0;
                        if (dist < minDistance && dist < 120) {
                            minDistance = dist;
                            closest = feat;
                        }
                    } catch (e) {}
                }
                found = closest;
            }

            if (found && !matchedPolygons.some(p => p.properties.id === found.properties.id)) {
                matchedPolygons.push(found);
            }
        });

        if (matchedPolygons.length === 0) {
            dom.dockStatusHint.innerText = 'Nenhum piquete identificado nesta coordenada. Clique no interior de uma pastagem.';
            dom.btnConfirmPaddock.disabled = true;
            dom.btnConfirmPaddock.classList.remove('ready');
            return;
        }

        // 2. Une os polígonos dos pontos positivos se houver múltiplos
        let unionPoly = matchedPolygons[0];
        for (let i = 1; i < matchedPolygons.length; i++) {
            try {
                const merged = turf.union(unionPoly, matchedPolygons[i]);
                if (merged) unionPoly = merged;
            } catch (e) {
                console.warn("Falha na união de polígonos:", e);
            }
        }

        // 3. Aplica pontos negativos (−) como zonas de exclusão/recorte
        if (negPoints.length > 0) {
            negPoints.forEach(np => {
                try {
                    // Buffer circular de 90m ao redor do ponto negativo
                    const negCircle = turf.circle([np.lon, np.lat], 0.09, { units: 'kilometers', steps: 32 });
                    const diff = turf.difference(unionPoly, negCircle);
                    if (diff) {
                        unionPoly = diff;
                    }
                } catch (e) {
                    console.warn("Falha no recorte negativo:", e);
                }
            });
        }

        // 4. Calcula métricas finais
        const area_m2 = turf.area(unionPoly);
        const area_ha = Math.max(0.1, Number((area_m2 / 10000.0).toFixed(2)));
        const baseScore = matchedPolygons[0].properties.confidence_score || 94.5;
        const confidence_score = Number(Math.min(99.2, Math.max(88.0, baseScore + (posPoints.length * 0.8) - (negPoints.length * 0.5))).toFixed(1));

        state.candidateMask = {
            area_ha,
            confidence_score,
            geojson: unionPoly.geometry
        };

        // 5. Renderiza Contorno Neon Tracejado Ciano no Leaflet
        if (state.candidateLayer) {
            map.removeLayer(state.candidateLayer);
        }

        state.candidateLayer = L.geoJSON(unionPoly, {
            style: {
                color: '#06b6d4',          // Ciano neon
                weight: 3,
                dashArray: '6, 6',
                fillColor: '#10b981',      // Esmeralda
                fillOpacity: 0.38
            }
        }).addTo(map);

        // Atualiza Badges e Botões
        dom.candidateScoreBadge.innerText = `Confiança: ${confidence_score}%`;
        dom.candidateScoreBadge.style.display = 'inline-block';
        dom.candidateAreaBadge.innerText = `${area_ha} ha`;
        dom.candidateAreaBadge.style.display = 'inline-block';

        dom.btnConfirmPaddock.disabled = false;
        dom.btnConfirmPaddock.classList.add('ready');

        if (state.editingPaddockId) {
            dom.dockTitleText.innerHTML = '<i data-lucide="edit-3"></i> Refinando Piquete Existente';
            dom.dockStatusHint.innerText = `Área calculada: ${area_ha} ha. Ajuste pontos ou clique em 'Atualizar Piquete'.`;
        } else {
            dom.dockTitleText.innerHTML = '<i data-lucide="crosshair"></i> Rascunho de Piquete Proposto';
            dom.dockStatusHint.innerText = `Polígono gerado: ${area_ha} ha (${confidence_score}%). Confirme para salvar!`;
        }

        if (window.lucide) lucide.createIcons();
    }

    // =========================================================================
    // Desfazer & Cancelar
    // =========================================================================
    dom.btnUndoPoint.addEventListener('click', () => {
        if (state.candidatePoints.length === 0) return;
        const last = state.candidatePoints.pop();
        if (last && last.marker) map.removeLayer(last.marker);
        refreshCandidatePrediction();
    });

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

        state.currentMode = 'positive';
        dom.btnModePositive.classList.add('active');
        dom.btnModeNegative.classList.remove('active');

        if (window.lucide) lucide.createIcons();
    }

    // =========================================================================
    // Confirmação de Piquete (Salva no Estado Local)
    // =========================================================================
    dom.btnConfirmPaddock.addEventListener('click', () => {
        if (!state.candidateMask) return;

        const pointsData = state.candidatePoints.map(p => ({ lon: p.lon, lat: p.lat, label: p.label }));

        if (state.editingPaddockId) {
            // Atualização de piquete existente
            const pIdx = state.savedPaddocks.findIndex(p => p.id === state.editingPaddockId);
            if (pIdx !== -1) {
                const current = state.savedPaddocks[pIdx];
                if (current.layer) map.removeLayer(current.layer);

                current.area_ha = state.candidateMask.area_ha;
                current.confidence_score = state.candidateMask.confidence_score;
                current.geojson = state.candidateMask.geojson;
                current.points = pointsData;
                current.layer = createPaddockMapLayer(current);

                clearCandidateSession();
                updatePaddocksStats();
                renderPaddocksList();
            }
        } else {
            // Criação de novo piquete
            const paddockNumber = state.savedPaddocks.length + 1;
            const newPaddock = {
                id: `paddock_${Date.now()}`,
                name: `Piquete ${paddockNumber}`,
                area_ha: state.candidateMask.area_ha,
                confidence_score: state.candidateMask.confidence_score,
                geojson: state.candidateMask.geojson,
                points: pointsData
            };

            newPaddock.layer = createPaddockMapLayer(newPaddock);
            state.savedPaddocks.push(newPaddock);

            clearCandidateSession();
            updatePaddocksStats();
            renderPaddocksList();
        }
    });

    /**
     * Cria a camada Leaflet definitiva com estilo verde esmeralda e popup
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
     * Modo de edição de piquete já salvo
     */
    function startEditPaddock(id) {
        const p = state.savedPaddocks.find(item => item.id === id);
        if (!p) return;

        clearCandidateSession();

        state.editingPaddockId = p.id;
        dom.confirmBtnLabel.innerText = "Atualizar Piquete";

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
                marker.on('click', (ev) => {
                    L.DomEvent.stopPropagation(ev);
                    removePromptPoint(pointId);
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

        state.savedPaddocks.forEach((p) => {
            const card = document.createElement('div');
            card.className = 'paddock-card';
            card.innerHTML = `
                <div class="paddock-info">
                    <span class="paddock-name">${p.name}</span>
                    <div class="paddock-meta">
                        <span class="paddock-meta-badge">${p.area_ha} ha</span>
                        <span>•</span>
                        <span>${p.confidence_score}%</span>
                    </div>
                </div>
                <div class="paddock-card-actions">
                    <button class="btn-card-action btn-edit-paddock" title="Refinar este Piquete">
                        <i data-lucide="edit-3"></i>
                    </button>
                    <button class="btn-card-action btn-delete-paddock" title="Excluir Piquete">
                        <i data-lucide="trash-2"></i>
                    </button>
                </div>
            `;

            // Clique no card centraliza o piquete no mapa
            card.addEventListener('click', (e) => {
                if (e.target.closest('button')) return;
                if (p.layer) {
                    map.fitBounds(p.layer.getBounds(), { padding: [60, 60], maxZoom: 18 });
                    p.layer.openPopup();
                }
            });

            // Botão Refinar
            card.querySelector('.btn-edit-paddock').addEventListener('click', (e) => {
                e.stopPropagation();
                startEditPaddock(p.id);
            });

            // Botão Excluir
            card.querySelector('.btn-delete-paddock').addEventListener('click', (e) => {
                e.stopPropagation();
                if (confirm(`Excluir ${p.name}?`)) {
                    if (p.layer) map.removeLayer(p.layer);
                    state.savedPaddocks = state.savedPaddocks.filter(item => item.id !== p.id);
                    updatePaddocksStats();
                    renderPaddocksList();
                }
            });

            dom.paddocksList.appendChild(card);
        });

        if (window.lucide) lucide.createIcons();
    }

    // Botão Limpar Todos os Piquetes
    dom.btnClearAllPaddocks.addEventListener('click', () => {
        if (state.savedPaddocks.length === 0) return;
        if (confirm("Deseja realmente limpar todos os piquetes mapeados?")) {
            state.paddocksLayer.clearLayers();
            state.savedPaddocks = [];
            clearCandidateSession();
            updatePaddocksStats();
            renderPaddocksList();
        }
    });

    // =========================================================================
    // Exportação de GeoJSON Client-Side (Download no Navegador)
    // =========================================================================
    dom.btnExportGeojson.addEventListener('click', () => {
        if (state.savedPaddocks.length === 0) return;

        const features = state.savedPaddocks.map(p => ({
            type: "Feature",
            properties: {
                id: p.id,
                name: p.name,
                area_ha: p.area_ha,
                confidence_score: p.confidence_score,
                points_count: p.points ? p.points.length : 0
            },
            geometry: p.geojson
        }));

        const featureCollection = {
            type: "FeatureCollection",
            features: features
        };

        const str = JSON.stringify(featureCollection, null, 2);
        const blob = new Blob([str], { type: "application/geo+json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        const carName = state.activePreset ? state.activePreset.id : "demonstracao";
        a.download = `piquetes_hq_sam_${carName}.geojson`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    // =========================================================================
    // Auxiliares de Loading Overlay
    // =========================================================================
    function showLoading(title, desc) {
        dom.loadingTitle.innerText = title;
        dom.loadingDesc.innerText = desc;
        dom.loadingOverlay.style.display = 'flex';
        setLoadingProgress(10);
    }

    function setLoadingProgress(percent) {
        dom.loadingBarFill.style.width = `${percent}%`;
    }

    function hideLoading() {
        dom.loadingOverlay.style.display = 'none';
        setLoadingProgress(0);
    }

    // =========================================================================
    // Inicialização da Aplicação
    // =========================================================================
    initPresets();
});
