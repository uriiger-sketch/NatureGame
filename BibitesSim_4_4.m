function BibitesSim_Advanced(rngSeed)
% BibitesSim_Advanced – Herbivore + Carnivore evolutionary world
% Optimized, vectorized, with selectable brains & predator-prey eating.
%
% Click a creature in the world to view its brain & stats in the "Clicked"
% panel (bottom-right). Red = carnivore. Others colored by genes.
%
% Uri build 2025-07-16

%% ─── reproducible RNG ─────────────────────────────────────────────────
if nargin==0, rngSeed = 1; end
rng(rngSeed,'twister');

%% ─── tunables ──────────────────────────────────────────────────────────
P = struct( ...
    'N0',30,            ... % initial herbivores
    'Npred',6,          ... % initial carnivores
    'worldSize',120,    ...
    'foodCount',80,    ...
    'foodEnergy',22,    ...
    'predEnergyGain',-10, ...       % energy predator gets per kill
    'bodyRadius',1.4,   ...
    'dt',1/5,          ... % 0.1 s/tick
    'maxSpeed',10,      ...
    'energyDecay',0.1, ...
    'moveCost',0.05,    ...
    'predMoveCost',0.07,... % extra cost for predator movement
    'reproEnergy',100,   ...
    'mutationSigma',0.3, ...
    'nnHidden',8,       ...
    'graphicsStride',2, ...
    'pauseRender',0.0,  ...
    'senseRadius',25,   ...
    'addLayerProb',0.18,...
    'delLayerProb',0.02,...
    'addNeuronProb',0.22,...
    'maxHiddenLayers',5,...
    'maxNeurons',70,    ...
    'fitFood',5,        ...
    'fitTick',0.01,     ...
    'fitRepro',20,       ...
    'maxAgeHerb', 1500, ...   % ≈20 min at dt = 0.1 s
    'maxAgeCarn',  500, ...   % carnivores die younger
'modeThresh', 0.10);   % |mode| ≤ 0.33 ⇒ freeze

% constants for NN I/O sizes
P.nInput = 5;           % [dx dy dNorm energyNorm bias]
P.nOutput = 3;          % [turn thrust reproduce]



%% ─── world state ──────────────────────────────────────────────────────
C = initCreatures(P);                  % initial herbivores
C = addCarnivores(C,P);                % add predators
baseW = C(1).W;                        % reference brain for diversity
F = rand(P.foodCount,2)*P.worldSize;   % food pellet positions
tick = 0;

% GUI selection state
overlayID = []; overlayH = []; overlayHalo = [];
overlayIDs = []; brainH = {}; haloH = {};

%% ─── graphics setup ───────────────────────────────────────────────────
figW = figure('Name','Bibites World','Color','k','Renderer','opengl');
axW = axes(figW,'Color',[.04 .04 .05], ...
           'XLim',[0 P.worldSize],'YLim',[0 P.worldSize], ...
           'DataAspectRatio',[1 1 1],'NextPlot','add');
axis(axW,'off');
title(axW,'');


%% ─── graphics setup ───────────────────────────────────────────────────
%figW = figure('Name','Bibites World','Color','k','Renderer','opengl');
% axW  = axes(figW, ...  % <‑‑ existing world axes
%             'Color',[.04 .04 .05], ...
%             'XLim',[0 P.worldSize],'YLim',[0 P.worldSize], ...
%             'DataAspectRatio',[1 1 1],'NextPlot','add');
% axis(axW,'off');

% ── NEW: live population plot (top‑left corner of same window) ─────────
axPop = axes('Parent',figW, ...
             'Position',[0.3 0.00 0.50 0.1], ...  % small strip
             'Color','none','NextPlot','add', ...
             'XColor','w','YColor','w','Box','on');
ylabel(axPop,'Count','Color','w','FontSize',7);

popCarnLine = plot(axPop,NaN,NaN,'r-','LineWidth',1);  % carnivores
popHerbLine = plot(axPop,NaN,NaN,'g-','LineWidth',1);  % herbivores
popTotLine  = plot(axPop,NaN,NaN,'w-','LineWidth',1);  % total

popT = [];   % time axis (ticks)
popC = [];   % carnivore count
popH = [];   % herbivore count


foodPlot = scatter(axW,F(:,1),F(:,2),16,'filled', ...
                   'MarkerFaceColor',[.2 .9 .2], ...
                   'MarkerEdgeColor','none');

% Sidebar brain panels
brainAx = gobjects(1,4);
x0 = 0.85; w = 0.18; h = 0.24; yTop = 0.72;
titles = {'#1','#2','#3','Clicked'};
for k=1:4
    brainAx(k) = axes('Parent',figW, ...
        'Position',[x0 yTop-(k-1)*h  w  h], ...
        'Color','none','XColor','none','YColor','none','NextPlot','add');
    title(brainAx(k),titles{k},'Color','w','FontSize',9,'Interpreter','none');
end

%% --- LEFT‑SIDE brain panels for TOP‑FIT creatures ---------------------
fitAx = gobjects(1,3);
xFit = 0.02;           % left margin
wFit = 0.18;           % same width
hFit = 0.24;           % same height
yTopFit = 0.72;
for k = 1:3
    fitAx(k) = axes('Parent',figW, ...
        'Position',[xFit yTopFit-(k-1)*hFit  wFit  hFit], ...
        'Color','none','XColor','none','YColor','none','NextPlot','add');
    title(fitAx(k), sprintf('Fit #%d',k), ...
          'Color','w','FontSize',9,'Interpreter','none');
end



% draw initial creature patches
for k=1:numel(C)
    C(k).patch = drawPatch(axW,C(k),P);
end

% GUI callbacks
set(figW,'WindowButtonDownFcn',@pickCreature);
uicontrol(figW,'Style','push','String','Save state',...
          'Units','normalized','Position',[.02 .02 .1 .05],...
          'Callback',@(~,~)saveState());
uicontrol(figW,'Style','push','String','Load state',...
          'Units','normalized','Position',[.14 .02 .1 .05],...
          'Callback',@(~,~)loadState());

%% ─── main loop ────────────────────────────────────────────────────────
while ishghandle(figW)
    tick = tick + 1;
    [C, F] = simStep(C, F, P);        % physics & ecology & evolution
    if ~isempty(C)
        lastPop = C; %#ok<NASGU> % (archival; not used here but kept)
    end

    % update graphics
    if mod(tick,P.graphicsStride)==0
[overlayH, overlayHalo, brainH, haloH] = ...
    syncGraphics(C, F, axW, foodPlot, P, baseW, ...
                 overlayID, overlayH, overlayHalo, ...
                 brainH, haloH, brainAx, fitAx, rngSeed, tick);

        pause(P.pauseRender);
    end
    if mod(tick,P.graphicsStride)==0
        % … existing syncGraphics call …

        % ── NEW: append and update population lines ───────────────────
        popT(end+1) = tick;
        popC(end+1) = sum(strcmp({C.type},'carn'));
        popH(end+1) = sum(strcmp({C.type},'herb'));
        set(popCarnLine,'XData',popT,'YData',popC);
        set(popHerbLine,'XData',popT,'YData',popH);
        set(popTotLine ,'XData',popT,'YData',popC+popH);
        xlim(axPop,[max(0,tick-1000) tick+1]);      % 1000‑tick sliding window
        ylim(axPop,[0 max(2,max(popC+popH))*1.1]);  % auto‑scale
        % ----------------------------------------------------------------
        pause(P.pauseRender);
    end

    if isempty(C)
        disp('All creatures extinct – simulation finished.');
        break
    end
end

%% ── nested callbacks ──────────────────────────────────────────────────
    function pickCreature(~, ~)
        % user clicked somewhere in world axes
        if isempty(C), return; end
        cp = get(axW,'CurrentPoint'); posClick = cp(1,1:2);
        % find nearest living creature to click
        pos = [[C.posX]' [C.posY]'];
        d2 = sum((pos - posClick).^2,2);
        [dMin, idxMin] = min(d2);
        if dMin < (P.bodyRadius*3)^2  % clickable radius
            overlayID = idxMin;
        else
            overlayID = [];
        end
    end

    function saveState()
        [f,p] = uiputfile('bibites_state.mat','Save state');
        if isequal(f,0), return; end
        save(fullfile(p,f),'C','F','tick','rngSeed','P');
    end

    function loadState()
        [f,p] = uigetfile('*.mat','Load state');
        if isequal(f,0), return; end
        S = load(fullfile(p,f));
        if isfield(S,'C'), C=S.C; end
        if isfield(S,'F'), F=S.F; end
        if isfield(S,'tick'), tick=S.tick; end
        if isfield(S,'rngSeed'), rngSeed=S.rngSeed; rng(rngSeed); end
        if isfield(S,'P'), P=S.P; end
        % ensure every loaded creature has patch + new fields
        for k=1:numel(C)
            if ~isfield(C(k),'patch'),    C(k).patch=[]; end
            if ~isfield(C(k),'type'),     C(k).type='herb'; end
            if ~isfield(C(k),'busyTime'), C(k).busyTime=0; end
            if ~isfield(C(k),'fadeTick'), C(k).fadeTick=0; end
            if ~isfield(C(k),'fadeInit'), C(k).fadeInit=0; end
        end
        % redraw everything immediately
        delete(findobj(axW,'Type','patch','-and','Tag','bibite'));
        for k=1:numel(C)
            C(k).patch = drawPatch(axW,C(k),P);
        end
[overlayH, overlayHalo, brainH, haloH] = ...
    syncGraphics(C, F, axW, foodPlot, P, baseW, ...
                 overlayID, overlayH, overlayHalo, ...
                 brainH, haloH, brainAx, fitAx, rngSeed, tick);

    end
end % BibitesSim_Advanced main


%% ======================================================================
%                              HELPERS
% ======================================================================

function C = initCreatures(P)
% initial herbivores
    C = repmat(struct(), P.N0, 1);
    for k=1:P.N0
        C(k).posX = rand*P.worldSize;
        C(k).posY = rand*P.worldSize;
        C(k).velX = 0; C(k).velY = 0;
        C(k).angle = rand*2*pi;
        % initial brain: one hidden layer
        C(k).W = {randn(P.nnHidden,P.nInput)*0.5; randn(P.nOutput,P.nnHidden)*0.5};
        C(k).energy = 55 + rand*12;
        C(k).act = zeros(P.nOutput,1);
        C(k).patch = [];
        C(k).color = geneColor(C(k));
        C(k).age = 0;
        C(k).fitness = 30;
        C(k).type = 'herb';          % herbivore
        C(k).busyTime = 0;           % predator only, but keep field
        C(k).fadeTick = 0;           % being eaten visual
        C(k).fadeInit = 0;
        C(k).immature = 0;              % adults initialised at 0
        C(k).lineage = k;      %  <<< one unique integer per founder
    end
end

function C = addCarnivores(C,P)
% append Npred carnivores (red)
    n0 = numel(C);
    for k=1:P.Npred
        i = n0 + k;
        C(i).posX = rand*P.worldSize;
        C(i).posY = rand*P.worldSize;
        C(i).velX = 0; C(i).velY = 0;
        C(i).angle = rand*2*pi;
        C(i).W = {randn(P.nnHidden,P.nInput)*0.5; randn(P.nOutput,P.nnHidden)*0.5};
        C(i).energy = 60 + rand*20;
        C(i).act = zeros(P.nOutput,1);
        C(i).patch = [];
        C(i).color = [1 0 0];
        C(i).age = 0;
        C(i).fitness = 30;
        C(i).type = 'carn';
        C(i).busyTime = 0;
        C(i).fadeTick = 0;
        C(i).fadeInit = 0;
        C(i).immature = 0;              % adults initialised at 0
        C(i).lineage = i;      %  <<< one unique integer per founder
    end
end


%% ───────────────────────── Simulation Step ────────────────────────────
function [C, F] = simStep(C, F, P)
% ---- SAFE extraction of position / velocity / etc. -------------------
% ----------------------------------------------------------------------
% SAFE extraction of bibite state (never drops a creature)
% ----------------------------------------------------------------------
% ----------------------------------------------------------------------
% Robust extraction of creature state ― never drops or deletes a row
% ----------------------------------------------------------------------
n = numel(C);

pos = zeros(n,2);
vel = zeros(n,2);
ang = zeros(n,1);
E   = zeros(n,1);
fit = zeros(n,1);
age = zeros(n,1);

for k = 1:n
    % guarantee numeric scalar in every field
    if isempty(C(k).posX),     C(k).posX = rand * P.worldSize; end
    if isempty(C(k).posY),     C(k).posY = rand * P.worldSize; end
    if isempty(C(k).velX),     C(k).velX = 0; end
    if isempty(C(k).velY),     C(k).velY = 0; end
    if isempty(C(k).angle),    C(k).angle = rand*2*pi; end
    if isempty(C(k).energy),   C(k).energy = P.foodEnergy; end
    if isempty(C(k).fitness),  C(k).fitness = 0; end
    if isempty(C(k).age),      C(k).age = 0; end

    pos(k,:) = [C(k).posX  C(k).posY];
    vel(k,:) = [C(k).velX  C(k).velY];
    ang(k)   =  C(k).angle;
    E(k)     =  C(k).energy;     % << never []
    fit(k)   =  C(k).fitness;    % << use *fit*, not “fitness”
    age(k)   =  C(k).age;
end

% make sure they are column vectors (prevents size‑mismatch later)
E   = E(:);          % ensure column orientation
fit = fit(:);        %      »        »

% ----------------------------------------------------------------------


% ----------------------------------------------------------------------

    if n == 0
        F = spawnFood(F,P);
        return;
    end

    % Extract state arrays
    busy = [C.busyTime]';
    fade = [C.fadeTick]';

    % Age & fitness
    age = age + 1;
    fit = fit + P.fitTick;

fit = fit(:);        %      »        »
    % decrement busy timers (predators digesting)
    busy = max(0, busy - 1);

    % decrement fade timers (prey shrinking)
    fade = max(0, fade - 1);



% ------------------------------------------------------------------
%                S E N S I N G
% ------------------------------------------------------------------
dx = zeros(n,1);                       % signed offset X‑to‑target
dy = zeros(n,1);                       %                 Y‑…
d  = P.senseRadius * ones(n,1);        % distance (clamped to radius)

isCarn = strcmp({C.type}','carn')';    % logical row‑vector → make column
isCarn = isCarn(:);
isHerb = ~isCarn;

% ===== Herbivore: food unless predator in sight ==================
if any(isHerb)
    idxHerb = find(isHerb);

    % ---------- nearest food vector ----------
    if ~isempty(F)
        DDx = pos(idxHerb,1) - F(:,1)';   DDx = torus(DDx,P.worldSize);
        DDy = pos(idxHerb,2) - F(:,2)';   DDy = torus(DDy,P.worldSize);
        dist2 = DDx.^2 + DDy.^2;
        [minFood2, idxFood] = min(dist2,[],2);
        dxF = DDx(sub2ind(size(DDx),(1:numel(idxHerb))',idxFood));
        dyF = DDy(sub2ind(size(DDy),(1:numel(idxHerb))',idxFood));
    else
        dxF = zeros(numel(idxHerb),1);
        dyF = dxF;
        minFood2 = (P.senseRadius^2)*ones(numel(idxHerb),1);
    end

    % ---------- nearest predator (danger) ----------
    if any(isCarn)
        DDxP = pos(idxHerb,1) - pos(isCarn,1)'; DDxP = torus(DDxP,P.worldSize);
        DDyP = pos(idxHerb,2) - pos(isCarn,2)'; DDyP = torus(DDyP,P.worldSize);
        dist2P = DDxP.^2 + DDyP.^2;
        [minPred2, idxPred] = min(dist2P,[],2);
        dxP = DDxP(sub2ind(size(DDxP),(1:numel(idxHerb))',idxPred));
        dyP = DDyP(sub2ind(size(DDyP),(1:numel(idxHerb))',idxPred));
    else
        minPred2 = Inf*ones(numel(idxHerb),1);
    end

    % ---------- choose: flee predator OR go to food ----------
    alert = minPred2 < minFood2;                     % predator closer
    idxA  = idxHerb(alert);                          % herbs in danger
    idxB  = idxHerb(~alert);                         % safe herbs

    % flee ⇒ vector *away* from predator
    % dx(idxA) = -dxP(alert);      dy(idxA) = -dyP(alert);
    d(idxA)  = sqrt(minPred2(alert));

    % forage ⇒ vector toward food
    % dx(idxB) =  dxF(~alert);     dy(idxB) =  dyF(~alert);
    d(idxB)  = sqrt(minFood2(~alert));
end

% ===== Carnivore: nearest weaker herbivore **not same lineage** =====
if any(isCarn)
    idxCarn = find(isCarn);

    % ------- make sure every creature has .lineage ------------------
    %   (safety net – older save‑states may lack it)
    if ~isfield(C,'lineage')
        [C.lineage] = deal(randi(1e9));      % random family tags
    end
    lineage = [C.lineage]';                  % column vector

    for cc = idxCarn.'             % iterate over real indices
        % Candidate prey: herbivore, not fading,
        %                 weaker AND different lineage
        preyMask = isHerb & fade==0 & ...
                   (E < E(cc)) & (fit < fit(cc)) & ...
                   (lineage ~= lineage(cc));       % ! lineage veto

        if ~any(preyMask),  continue,  end

        preyIdx = find(preyMask);               % indices of all edible
        ddx = torus(pos(preyIdx,1) - pos(cc,1), P.worldSize);
        ddy = torus(pos(preyIdx,2) - pos(cc,2), P.worldSize);
        [bestD, jj] = min( hypot(ddx, ddy) );

        if bestD > P.senseRadius,  continue,  end

        dx(cc) = ddx(jj);                       % set hunt‑vector
        dy(cc) = ddy(jj);
        d(cc)  = bestD;
    end
end

% ------------------------------------------------------------------
%  ✱  Herbivores run away from the nearest carnivore in sight
% ------------------------------------------------------------------
if any(isHerb) && any(isCarn)
    H   = find(isHerb);              % herbivores
    Cp  = find(isCarn);              % predators

    % signed toroidal offsets herb → all carnivores
    dX =  torus( pos(H,1) - pos(Cp,1)', P.worldSize);   % |H| × |C|
    dY =  torus( pos(H,2) - pos(Cp,2)', P.worldSize);

    dist2 = dX.^2 + dY.^2;
    [minD2, nearest] = min(dist2, [], 2);               % one predator / herb

    alert = minD2 < (P.senseRadius*P.senseRadius);      % within sight radius
    if any(alert)
        idxA  = H(alert);           % indices of alerted herbivores
        targ  = Cp(nearest(alert)); % their corresponding predator indices

        % vector FROM predator TO herbivore
        dxPred = torus( pos(idxA,1) - pos(targ,1), P.worldSize );
        dyPred = torus( pos(idxA,2) - pos(targ,2), P.worldSize );

        % overwrite the sensing vector so the herb flees
        dx(idxA) = dxPred;
        dy(idxA) = dyPred;
        d(idxA)  = sqrt( dxPred.^2 + dyPred.^2 );
    end
end

% ---------- clamp outside sight radius ----------
maskFar = d > P.senseRadius;
dx(maskFar)=0; dy(maskFar)=0; d(maskFar)=P.senseRadius;
% ------------------------------------------------------------------
%  ✱  Herbivores run away from the nearest carnivore in sight
% ------------------------------------------------------------------
if any(isHerb) && any(isCarn)
    H   = find(isHerb);              % herbivores
    Cp  = find(isCarn);              % predators

    dX = torus(pos(H,1) - pos(Cp,1)', P.worldSize);   % |H| × |C|
    dY = torus(pos(H,2) - pos(Cp,2)', P.worldSize);
    dist2 = dX.^2 + dY.^2;

    [minD2, nearest] = min(dist2, [], 2);             % one predator / herb
    alert = minD2 < P.senseRadius^2;                  % within sight radius
    if any(alert)
        idxA  = H(alert);           % alerted herbivores
        targ  = Cp(nearest(alert)); % matching predators

        % vector FROM predator TO herbivore  (direction to flee)
        dxPred = torus(pos(idxA,1) - pos(targ,1), P.worldSize);
        dyPred = torus(pos(idxA,2) - pos(targ,2), P.worldSize);

        dx(idxA) = dxPred;          % overwrite sensing vector
        dy(idxA) = dyPred;
        d(idxA)  = sqrt(dxPred.^2 + dyPred.^2);
    end
end


% % % % % % % % % % % % % ------------------------------------------------------------------
% % % % % % % % % % % % % SENSING  (herbivores → nearest food pellet,
% % % % % % % % % % % % %           carnivores → nearest edible prey)
% % % % % % % % % % % % % Returns column vectors dx,dy (signed toroidal offsets) and d (true distance)
% % % % % % % % % % % % % ------------------------------------------------------------------
% % % % % % % % % % % % dx = zeros(n,1);  dy = zeros(n,1);  d  = P.senseRadius * ones(n,1);
% % % % % % % % % % % % 
% % % % % % % % % % % % isCarn = strcmp({C.type}','carn');
% % % % % % % % % % % % isCarn = isCarn(:);
% % % % % % % % % % % % isHerb = ~isCarn;
% % % % % % % % % % % % 
% % % % % % % % % % % % % ---------- Herbivores: target food pellets -----------------------
% % % % % % % % % % % % if any(isHerb) && ~isempty(F)
% % % % % % % % % % % %     H = find(isHerb);
% % % % % % % % % % % %     ddX = pos(H,1) - F(:,1)';  ddX = torus(ddX, P.worldSize);
% % % % % % % % % % % %     ddY = pos(H,2) - F(:,2)';  ddY = torus(ddY, P.worldSize);
% % % % % % % % % % % %     dist2 = ddX.^2 + ddY.^2;
% % % % % % % % % % % %     [minDist2, idxFood] = min(dist2, [], 2);
% % % % % % % % % % % % 
% % % % % % % % % % % %     dx(H) = ddX(sub2ind(size(ddX), (1:numel(H))', idxFood));
% % % % % % % % % % % %     dy(H) = ddY(sub2ind(size(ddY), (1:numel(H))', idxFood));
% % % % % % % % % % % %     d(H)  = sqrt(minDist2);
% % % % % % % % % % % % end
% % % % % % % % % % % % 
% % % % % % % % % % % % % ---------- Carnivores: target live, edible prey ------------------
% % % % % % % % % % % % % ---------- Carnivores: target live, edible prey ------------------
% % % % % % % % % % % % % ---------- Carnivores: target live, edible prey ------------------
% % % % % % % % % % % % if any(isCarn)
% % % % % % % % % % % %     Cidx = find(isCarn);
% % % % % % % % % % % %     for cc = Cidx.'
% % % % % % % % % % % % 
% % % % % % % % % % % %         % ‑‑ edible if herb, OR weaker carnivore
% % % % % % % % % % % %         herbMask = ~isCarn;                                 % n×1
% % % % % % % % % % % %         sameLineage = [C.lineage]' == C(i).lineage;     % column‑logical
% % % % % % % % % % % %         weakerCarn =  (E < E(cc)) & (fit < fit(cc) & ~sameLineage);        % n×1
% % % % % % % % % % % %         carnMask  =  isCarn & weakerCarn;                   % n×1
% % % % % % % % % % % % 
% % % % % % % % % % % %         preyMask = (herbMask | carnMask) & ...              % right species
% % % % % % % % % % % %                    fade == 0          & ...                 % not fading
% % % % % % % % % % % %                    (1:n).' ~= cc;                           % not self
% % % % % % % % % % % % 
% % % % % % % % % % % %         if ~any(preyMask), continue, end      % nothing edible
% % % % % % % % % % % % 
% % % % % % % % % % % %         % find nearest prey
% % % % % % % % % % % %         preyIdx = find(preyMask);              % linear column
% % % % % % % % % % % %         ddx = torus(pos(preyIdx,1) - pos(cc,1), P.worldSize);
% % % % % % % % % % % %         ddy = torus(pos(preyIdx,2) - pos(cc,2), P.worldSize);
% % % % % % % % % % % %         [minD, j] = min( hypot(ddx, ddy) );
% % % % % % % % % % % % 
% % % % % % % % % % % %         if minD > P.senseRadius, continue, end
% % % % % % % % % % % % 
% % % % % % % % % % % %         dx(cc) = ddx(j);
% % % % % % % % % % % %         dy(cc) = ddy(j);
% % % % % % % % % % % %         d(cc)  = minD;
% % % % % % % % % % % %     end
% % % % % % % % % % % % end

% ======  BRAIN FORWARD PASS  ==========================================
% (recreates variable "act")
act = zeros(n, P.nOutput);          % n creatures  ×  3 outputs
for i = 1:n
    % skip if creature is immobilised (digesting or being eaten)
    if busy(i) > 0 || fade(i) > 0,  continue,  end

    inp = [ dx(i);                      % x‑offset of target
            dy(i);                      % y‑offset of target
            d(i) / P.senseRadius;       % distance  (0‑1)
            E(i) / P.reproEnergy;       % energy    (0‑1+)
            1 ];                        % bias

    x = inp;
    for L = 1:numel(C(i).W)-1           % hidden layers
        x = tanh( C(i).W{L} * x );
    end
    act(i,:) = tanh( C(i).W{end} * x )';    % outputs: [mode thrust reproduce]
end
% ======================================================================

% ---------- Clip to sensing radius --------------------------------
maskFar = d > P.senseRadius;
dx(maskFar) = 0;   dy(maskFar) = 0;   d(maskFar) = P.senseRadius;


   % -------------------------------------------------- INTERPRET OUTPUT ----
modeRaw  = act(:,1);                          % new selector
thrustIn = max(act(:,2),0);                   % keep 2nd output
mode     = zeros(n,1);                        % –1 flight, 0 freeze, +1 fight
mode(modeRaw >  P.modeThresh)  =  1;
mode(modeRaw < -P.modeThresh)  = -1;
% outputs: [mode  thrust  reproduce]
%   mode  <‑0.33   ⇢  FLIGHT  (run away)
%         |mode|≤0.33 ⇢  FREEZE (no thrust)
%         mode > 0.33 ⇢  FIGHT  (approach & bite/eat)
% ---------------------------------------- choose target direction ------
% vector toward the same pellet (herb) or prey centre (carn) – we re‑use dx,dy
thetaTarget = atan2(dy,dx);                   % radians to target
thetaTarget(mode==-1) = thetaTarget(mode==-1) + pi;   % FLIGHT ⤳ opposite

% rotate smoothly toward the target (small inertial turn)
dAng = wrapToPi(thetaTarget - ang);           % signed shortest diff
ang  = ang + 3*P.dt .* dAng;                  % 3 rad s⁻¹ turn‑rate cap

% thrust only if mode≠0
thrust = thrustIn .* (mode~=0);

vel(:,1) = vel(:,1) + thrust .* cos(ang);
vel(:,2) = vel(:,2) + thrust .* sin(ang);


    % no movement if busy digesting or fading (dead/being eaten)
    imm = (busy > 0) | (fade > 0);
    vel(imm,:) = 0;

    % clamp speed
    spd = sqrt(sum(vel.^2,2));
    too = spd > P.maxSpeed;
    if any(too)
        scl = P.maxSpeed ./ spd(too);
        vel(too,:) = vel(too,:) .* [scl scl];
        spd(too) = P.maxSpeed;
    end

    % integrate
    pos = mod(pos + vel * P.dt, P.worldSize);

    % energy decay + move cost
    E = E - P.energyDecay - P.moveCost .* spd;
    E   = E(:);          % ensure column orientation

    % extra cost for carnivores
isCarn = strcmp({C.type}','carn');
isCarn = isCarn(:);
isHerb = ~isCarn;

    E(isCarn) = E(isCarn) - P.predMoveCost .* spd(isCarn);

herbIdx = find(strcmp({C.type}','herb')' & fade==0);
herbIdx = herbIdx(herbIdx <= numel(E));   % safe length
if ~isempty(herbIdx)
    [E(herbIdx), F, eatCnt] = eatLoop(pos(herbIdx,:), F, E(herbIdx), P);
    fit(herbIdx) = fit(herbIdx) + eatCnt * P.fitFood;
end
E   = E(:);          % ensure column orientation
fit = fit(:);        %      »        »




    % --- Carnivore attacks
% --- Carnivore attacks -------------------------------------------------
for i = find(isCarn & busy==0 & fade==0).'          % each active predator
    % ------- assemble candidate list as a *column* logical -------------
    preyMask          = false(n,1);                 % start empty
    preyMask(~isCarn) = true;                       % every herbivore

    weakerCarn        =  isCarn  & (E <  E(i)) & (fit < fit(i));
    preyMask          =  preyMask | weakerCarn;     % add weaker carnivores

    % can’t eat itself or anything fading
    preyMask(i)      = false;
    preyMask(fade>0) = false;

    if ~any(preyMask),  continue,  end

    % ----------- choose nearest prey -----------------------------------
    preyIdx  = find(preyMask);                      % guaranteed ≤ n
    dxP      = torus(pos(preyIdx,1) - pos(i,1), P.worldSize);
    dyP      = torus(pos(preyIdx,2) - pos(i,2), P.worldSize);
    [minD,j] = min(hypot(dxP,dyP));
    if minD > 2*P.bodyRadius,  continue,  end

    prey = preyIdx(j);
    if prey > numel(E),  continue;  end        % prey vanished

    % ----- begin eating sequence --------------------------------------
    % when the predator bites
eatTicks        = 10;
busy(i)         = ceil(eatTicks);     % digest that time
fade(prey)      = eatTicks;             % prey still fades the full time
C(prey).fadeInit= eatTicks;
    E(i)            = E(i) + E(prey)/4;      % absorb prey energy
    fit(i)=fit(i)+P.fitFood;
end


    % --- OLD‑AGE deaths ------------------------------------------------
    isCarn = strcmp({C.type},'carn')';      % logical column
    isHerb = ~isCarn;

    deadOld = (isHerb & age >= P.maxAgeHerb) | ...
              (isCarn & age >= P.maxAgeCarn);

    % --- existing death conditions ------------------------------------
    deadFade   = (fade == 0 & [C.fadeInit]' > 0);  % eaten & finished fading
    deadEnergy = (E <= 0);

    dead = deadOld | deadFade | deadEnergy;   % <‑‑ just add the new mask


    % ------- keep survivors, BUT trim all vectors first ---------------
    keep = ~dead;

    pos  = pos(keep,:);  vel  = vel(keep,:);  ang  = ang(keep);
    E    = E(keep);      fit  = fit(keep);    age  = age(keep);
    busy = busy(keep);   fade = fade(keep);

    fadeInit = [C.fadeInit]';        % full-length copy
    fadeInit = fadeInit(keep);       % aligned

    C = C(keep);                     % shorten the struct array *last*
    n = numel(C);                    % new population size

    % % % % % --- kill faded-out prey when fade hits 0 OR energy <=0
    % % % % deadFade = (fade==0 & [C.fadeInit]' > 0); % those that were fading and finished
    % % % % deadEnergy = E <= 0;
    % % % % dead = deadFade | deadEnergy;
    % % % % 
    % % % % % filter survivors
    % % % % keep = ~dead;
    % % % % C = C(keep);
    % % % % pos = pos(keep,:); vel = vel(keep,:); ang = ang(keep);
    % % % % E   = E(keep); fit = fit(keep); age = age(keep);
    % % % % busy = busy(keep); fade = fade(keep);
    % % % % % update fadeInit for survivors
    % % % % fadeInitSurviv = [C.fadeInit]';
    % % % % fadeInitSurviv = fadeInitSurviv(keep);

    % --- write back
    for k=1:numel(C)
        C(k).posX = pos(k,1); C(k).posY = pos(k,2);
        C(k).velX = vel(k,1); C(k).velY = vel(k,2);
        C(k).angle = ang(k);
        C(k).energy = E(k);
        C(k).fitness = fit(k);
        C(k).age = age(k);
        C(k).act = act(k,:)';
        C(k).busyTime = busy(k);
        C(k).fadeTick = fade(k);
        % C(k).fadeInit = fadeInitSurviv(k);
        % color stays (carn red override in drawPatch)
    end

    % --- Reproduction (children inherit type)
    i = 1;
    while i <= numel(C)
        if C(i).fadeTick>0 || C(i).busyTime>0
            i=i+1; continue; % can't reproduce mid-fade/digest
        end
        if C(i).energy >= P.reproEnergy && C(i).act(3) > 0
            C = reproduce(C, i, P);
        end
        i = i + 1;
    end

    % --- keep food density
    F = spawnFood(F,P);
end


%% ───────────────────── Structural-preserving Reproduce ─────────────────
%% ───────────────────── Structural‑preserving Reproduce ─────────────────
function C = reproduce(C, i, P)
% Spawns a single child from parent C(i), with
% • energy split 50‑50
% • weight + structural mutations
% • “new‑born shield” → chi.immature = round(2 s / dt)
% • short protection for the parent too (so it can’t be eaten at once)

    par = C(i);                                  % ==== parent snapshot
    if ~isfield(par,'type'),  par.type = 'herb'; end

    %% --- energy & fitness bookkeeping ---------------------------------
    eHalf          = par.energy/2;
    par.energy     = eHalf;
    par.fitness    = par.fitness + P.fitRepro;

    %% --- build child ---------------------------------------------------
    chi            = par;                         % struct copy
    chi.energy     = eHalf;
    chi.fitness    = par.fitness/2;
    chi.age        = 0;
    chi.busyTime   = 0;
    chi.fadeTick   = 0;
    chi.fadeInit   = 0;

    % 2‑second invulnerability shield
    chi.immature   = round( 2 / P.dt );           % NEW  ←••
    % also give the parent a very brief shield (so it can’t be chain‑eaten)
    if ~isfield(par,'immature'),  par.immature = 0; end
    par.immature   = max(par.immature, 5);        % ≈0.5 s

    %--- mutate weights (value + structure) -----------------------------
    for L = 1:numel(chi.W)
        chi.W{L} = par.W{L} + randn(size(par.W{L}))*P.mutationSigma;
    end
    chi.W = mutateNetworkStructure(chi.W,P);

    %--- derive colour / pose ------------------------------------------
    if strcmp(chi.type,'carn')
        chi.color = [1 0 0];
    else
        chi.color = geneColor(chi);
    end
    chi.angle = rand*2*pi;
    chi.velX  = 0;  chi.velY = 0;
    chi.posX  = mod(par.posX + cos(par.angle)*P.bodyRadius*3, P.worldSize);
    chi.posY  = mod(par.posY + sin(par.angle)*P.bodyRadius*3, P.worldSize);
    chi.patch = [];

    %--- commit ---------------------------------------------------------
    C(i)      = par;
    C(end+1)  = chi;
end



%% ───────────────────── Network Structure Mutation ─────────────────────
function W = mutateNetworkStructure(W, P)
% W is 1..L+1 cells: hidden→..., output last
% robust safe ops (no invalid indexing)
    inSize  = P.nInput;
    outSize = P.nOutput;
    L = numel(W)-1;  % hidden count

    % maybe add hidden (insert before output)
    if L < P.maxHiddenLayers && rand < P.addLayerProb
        % width similar to last hidden
        nPrev = size(W{end-1},1);
        nNew  = max(2, min(P.maxNeurons, round(nPrev*(0.7+0.6*rand))));
        % build new hidden bridging matrix
        Wnew  = randn(nNew, nPrev) * 0.3;
        % resize output to accept nNew
        WoutNew = randn(outSize, nNew) * 0.3;
        W = [W(1:end-1) {Wnew} {WoutNew}];
        L = L + 1;
    end

    % maybe delete one hidden
    if L > 1 && rand < P.delLayerProb
        kill = randi(L); % which hidden
        if kill == 1
            prevW = inSize;
        else
            prevW = size(W{kill-1},1);
        end
        if kill == L
            % killing last hidden → new output directly from prev width
            W{end} = randn(outSize, prevW) * 0.3;
        else
            % killing middle hidden → rebuild next layer to take prev width
            nextRows = size(W{kill+1},1);
            W{kill+1} = randn(nextRows, prevW) * 0.3;
        end
        W(kill) = [];
        L = L - 1;
    end

    % maybe grow a neuron in random hidden
    if L >= 1 && rand < P.addNeuronProb
        which = randi(L);
        % enforce max
        if size(W{which},1) < P.maxNeurons
            colsPrev = size(W{which},2);              % input width to this hidden
            W{which}(end+1,:) = randn(1,colsPrev)*0.3; % add hidden neuron (new row)
            % add corresponding column to downstream matrix
            W{which+1}(:,end+1) = randn(size(W{which+1},1),1)*0.3;
        end
    end
end


%% ───────────────────── Eat-food loop (vectorized) ─────────────────────
function [E, F, eatCnt] = eatLoop(pos, F, E, P)
    m = size(pos,1);
    eatCnt = zeros(m,1);
    if isempty(F), return; end
    r2 = P.bodyRadius^2;
    dX = pos(:,1) - F(:,1)'; dX = torus(dX, P.worldSize);
    dY = pos(:,2) - F(:,2)'; dY = torus(dY, P.worldSize);
    hit = (dX.^2 + dY.^2) <= r2;
    eatCnt = sum(hit,2);
    eatenMask = any(hit,1);
    E = E + eatCnt * P.foodEnergy;
    F(eatenMask,:) = [];
end


%% ───────────────────── Gene-based Color (herb only) ───────────────────
function col = geneColor(cre)
    h = mod(mean(cre.W{1}(:))*0.4+0.5, 1);
    s = 0.6+0.3/(1+exp(-std(cre.W{end}(:))*2));
    col = hsv2rgb([h s 1]);
end


%% ───────────────────── Creature Polygon Verts ─────────────────────────
function [vx,vy] = polyVerts(cre, P)
% vary polygon complexity by NN signatures
    sig = tanh(sum(sign(cre.W{end}(:)))/numel(cre.W{end}));
    n = 3 + round((sig+1)*2.5);
    n = max(3, min(8, n));
    sizeG = 0.8 + 0.6/(1+exp(-std(cre.W{end}(:))*3));
    R = P.bodyRadius * sizeG;

    if cre.fadeTick > 0 && cre.fadeInit > 0
        R = R * (cre.fadeTick / cre.fadeInit);
    elseif cre.fadeInit > 0           % just finished fadin
        R = 0;                     % draw as a point → invisible
    end
    
    th = (0:n-1)'/n*2*pi + cre.angle;
    vx = cre.posX + R*cos(th);
    vy = cre.posY + R*sin(th);
end


%% ───────────────────── Food Respawn ───────────────────────────────────
function F = spawnFood(F, P)
    m = P.foodCount - size(F,1);
    if m > 0
        F = [F; rand(m,2)*P.worldSize];
    end
end


%% ───────────────────── Draw One Creature Patch ────────────────────────
function h = drawPatch(ax, cre, P)
    [vx, vy] = polyVerts(cre, P);
    if strcmp(cre.type,'carn')
        faceCol = [1 0 0];
    else
        faceCol = cre.color;
    end
    h = patch(ax, vx, vy, faceCol, ...
              'EdgeColor', 'none', ...
              'Tag', 'bibite');
end


%% ───────────────────── Update Creature Patch ──────────────────────────
function updateCreaturePatch(cre,P)
    [vx,vy] = polyVerts(cre,P);
    if strcmp(cre.type,'carn')
        col = [1 0 0];
    else
        col = cre.color;
    end
    set(cre.patch,'XData',vx,'YData',vy,'FaceColor',col);
end


%% ───────────────────── Graphics Sync ──────────────────────────────────
function [overlayH, overlayHalo, brainH, haloH] = ...
    syncGraphics(C, F, axW, foodPlot, P, baseW, ...
                 overlayID, overlayH, overlayHalo, ...
                 brainH, haloH, brainAx, fitAx, rngSeed, tick);

    % ensure each creature has patch
    for k=1:numel(C)
        if isempty(C(k).patch) || ~ishghandle(C(k).patch)
            C(k).patch = drawPatch(axW,C(k),P);
        end
    end

    % update food scatter
    if isempty(F)
        set(foodPlot,'XData',NaN,'YData',NaN);
    else
        set(foodPlot,'XData',F(:,1),'YData',F(:,2));
    end

    % remove patches that no longer correspond
    livePatches = [C.patch];
    allPatches  = findobj(axW,'Type','patch','-and','Tag','bibite');
    deadPatches = setdiff(allPatches, livePatches);
    if ~isempty(deadPatches), delete(deadPatches); end

    % update each creature patch
    for k=1:numel(C), updateCreaturePatch(C(k),P); end

    title(axW, sprintf('Tick %d  ·  Pop %d  ·  seed %d', ...
           tick, numel(C), rngSeed), 'Color','w','Interpreter','none');

    % rank diversity
    if ~isempty(C)
        scores = arrayfun(@(z) diversityScore(z, baseW), C);
        [~, ord] = sort(scores, 'descend');
        overlayIDs = ord(1:min(3,numel(ord)));
    else
        overlayIDs = [];
        scores = [];
    end

    % clear side panels
    for k=1:4, cla(brainAx(k)); end %#ok<AGROW>

    % draw top 3 diverse brains
    for k=1:numel(overlayIDs)
        idx = overlayIDs(k);
        plotNetwork(brainAx(k), C(idx));
        title(brainAx(k), sprintf('D %.2f  E %.1f  F %.0f  %s', ...
              scores(idx), C(idx).energy, C(idx).fitness, C(idx).type), ...
              'Color','w','FontSize',8);
    end

    % draw clicked creature
    if ~isempty(overlayID) && overlayID<=numel(C)
        plotNetwork(brainAx(4), C(overlayID));
        title(brainAx(4), sprintf('ID %d  %s  E %.1f  F %.1f  D %.2f  age %d', ...
              overlayID, C(overlayID).type, ...
              C(overlayID).energy, C(overlayID).fitness, ...
              diversityScore(C(overlayID), baseW), C(overlayID).age));
    end

    % halo highlight for clicked creature
    if ~isempty(overlayID) && overlayID <= numel(C)
        R  = P.bodyRadius * 2;
        th = linspace(0,2*pi,40);
        x  = C(overlayID).posX + R*cos(th);
        y  = C(overlayID).posY + R*sin(th);
        if isempty(overlayHalo) || ~ishghandle(overlayHalo)
            overlayHalo = plot(axW,x,y,'r','LineWidth',1.4,'Tag','clickedHalo');
        else
            set(overlayHalo,'XData',x,'YData',y);
        end
        % tiny network sketch just above
        overlayH = drawBrainOverlay(axW, C(overlayID), P, overlayH);
    else
        if ~isempty(overlayH)    && ishghandle(overlayH),    delete(overlayH);    end
        if ~isempty(overlayHalo) && ishghandle(overlayHalo), delete(overlayHalo); end
        overlayH = []; overlayHalo = [];
    end

    % halos for the 3 diverse
    need = numel(overlayIDs);
    brainH = extendHandles(brainH, need);
    haloH  = extendHandles(haloH,  need);
    for k = 1:need
        idx = overlayIDs(k);
        R = P.bodyRadius * 2;
        th = linspace(0,2*pi,40);
        x = C(idx).posX + R*cos(th);
        y = C(idx).posY + R*sin(th);
        if isempty(haloH{k}) || ~ishghandle(haloH{k})
            haloH{k} = plot(axW,x,y,'w','LineWidth',1.2,'Tag','diverseHalo');
        else
            set(haloH{k},'XData',x,'YData',y);
        end
        if ~isempty(brainH{k}) && ishghandle(brainH{k}), delete(brainH{k}); end
        brainH{k} = [];
    end
    brainH = pruneHandles(brainH,need);
    haloH  = pruneHandles(haloH, need);

    %%% --- TOP‑FITNESS ranking (left side) ---------------------------------
if ~isempty(C)
    fitScores = [C.fitness];
    [~, fitOrd] = sort(fitScores, 'descend');
    fitIDs = fitOrd(1:min(3,numel(fitOrd)));
else
    fitIDs = [];
end

% clear left‑side axes
for k = 1:3, cla(fitAx(k)); end

% draw brains for top‑fitness
for k = 1:numel(fitIDs)
    idx = fitIDs(k);
    plotNetwork(fitAx(k), C(idx));
    title(fitAx(k), sprintf('Fit %.0f  E %.1f', ...
          C(idx).fitness, C(idx).energy), ...
          'Color','w','FontSize',8);
end

    drawnow limitrate

    % local helpers
    function arr = extendHandles(arr,n)
        arr(end+1:n) = {[]};
    end
    function arr = pruneHandles(arr,nKeep)
        for p=(nKeep+1):numel(arr)
            if ishghandle(arr{p}), delete(arr{p}); end
        end
        arr = arr(1:nKeep);
    end
end


%% ───────────────────── Weight Color Mapping ───────────────────────────
% % % % function rgb = weightColor(w)
% % % %     w = max(-1, min(1, w)); w = w(:);
% % % %     if any(w<0)
% % % %         red   = max(0,  w);
% % % %         blue  = max(0,  w);
% % % %         green = max(0,  w);
% % % %         rgb   = [red green blue];
% % % %     else
% % % %         g = 1 - w;
% % % %         rgb = repmat(g, 1, 3);
% % % %     end
% % % % end
function rgb = weightColor(w)
    % Clamp to expected range
    w = max(-1, min(1, w));      % w ϵ [‑1,1]
    
    % Map ‑1 → 0 (black)   …   +1 → 1 (white)
    g = (w + 1) / 2;            % linear brightness
    
    % Return Nx3 RGB – every row identical, so colour is grey‑scale
    rgb = repmat(g(:), 1, 3);
end



%% ───────────────────── Mini Network Overlay ───────────────────────────
function overlayHandle = drawBrainOverlay(ax, cre, P, oldH)
    if ~isempty(oldH)&&ishghandle(oldH), delete(oldH); end
    yOff = 8;
    cy = mod(cre.posY + yOff, P.worldSize);
    cx = cre.posX;
    L = numel(cre.W)-1;
    xL = 0:L+1;
    X = @(idx) cx + idx*1.5;
    Y = @(v) cy + v;
    yIn  = linspace(-1, 1, P.nInput);
    yH   = arrayfun(@(k) linspace(-1,1,size(cre.W{k},1)), 1:L, 'UniformOutput',false);
    yOut = linspace(-.5, .5, P.nOutput);
    overlayHandle = hggroup(ax,'Tag','brainOverlay');
    lw = 1.2;
    % Input->first hidden
    if L>=1
        for i=1:numel(yIn)
            for j=1:numel(yH{1})
                w = cre.W{1}(j,i);
                line('Parent',overlayHandle,'LineWidth',lw,...
                     'XData',[X(xL(1)) X(xL(2))],...
                     'YData',[Y(yIn(i)) Y(yH{1}(j))],...
                     'Color',weightColor(w));
            end
        end
    end
    % Hidden->hidden
    for h=1:(L-1)
        W = cre.W{h+1};
        for i=1:numel(yH{h})
            for j=1:numel(yH{h+1})
                w = W(j,i);
                line('Parent',overlayHandle,'LineWidth',lw,...
                     'XData',[X(xL(h+1)) X(xL(h+2))],...
                     'YData',[Y(yH{h}(i)) Y(yH{h+1}(j))],...
                     'Color',weightColor(w));
            end
        end
    end
    % Last hidden->out
    if L>=1
        Wout = cre.W{end};
        for i=1:numel(yH{end})
            for j=1:numel(yOut)
                w = Wout(j,i);
                line('Parent',overlayHandle,'LineWidth',lw,...
                     'XData',[X(xL(end-1)) X(xL(end))],...
                     'YData',[Y(yH{end}(i)) Y(yOut(j))],...
                     'Color',weightColor(w));
            end
        end
    else
        % direct input->out (no hidden) case
        Wout = cre.W{end};
        for i=1:numel(yIn)
            for j=1:numel(yOut)
                w = Wout(j,i);
                line('Parent',overlayHandle,'LineWidth',lw,...
                     'XData',[X(0) X(1)],...
                     'YData',[Y(yIn(i)) Y(yOut(j))],...
                     'Color',weightColor(w));
            end
        end
    end
    % Nodes
    scatter(X(xL(1))*ones(1,numel(yIn)),  Y(yIn),  12,'k','filled','Parent',overlayHandle);
    for h=1:L
        scatter(X(xL(h+1))*ones(1,numel(yH{h})), Y(yH{h}), 14,'k','filled','Parent',overlayHandle);
    end
    scatter(X(xL(end))*ones(1,numel(yOut)), Y(yOut), 12,'k','filled','Parent',overlayHandle);
end


%% ───────────────────── Full Network Plot (sidebar) ────────────────────
function plotNetwork(ax, cre)
    cla(ax); hold(ax,'on'); axis(ax,'off','equal');
    L = numel(cre.W)-1; xL = 0:L+1;
    yIn = linspace(-1,1,size(cre.W{1},2)); % input width from W{1}
    yH = cell(1,L);
    for k=1:L
        yH{k} = linspace(-1,1,size(cre.W{k},1));
    end
    yOut = linspace(-0.5,0.5,size(cre.W{end},1));
    X = @(idx) xL(idx); Y = @(v) v;
    if L>=1
        % Input->first hidden
        for i=1:numel(yIn)
            for j=1:numel(yH{1})
                w = cre.W{1}(j,i);
                line(ax,[X(1) X(2)],[Y(yIn(i)) Y(yH{1}(j))],...
                     'Color',weightColor(w));
            end
        end
        % Hidden->hidden
        for h=1:(L-1)
            W = cre.W{h+1};
            for i=1:numel(yH{h})
                for j=1:numel(yH{h+1})
                    w = W(j,i);
                    line(ax,[X(h+1) X(h+2)],[Y(yH{h}(i)) Y(yH{h+1}(j))],...
                         'Color',weightColor(w));
                end
            end
        end
        % Last hidden->out
        Wout = cre.W{end};
        for i=1:numel(yH{end})
            for j=1:numel(yOut)
                w = Wout(j,i);
                line(ax,[X(L+1) X(L+2)],[Y(yH{end}(i)) Y(yOut(j))],...
                     'Color',weightColor(w));
            end
        end
    else
        % No hidden: Input->out
        Wout = cre.W{end};
        for i=1:numel(yIn)
            for j=1:numel(yOut)
                w = Wout(j,i);
                line(ax,[X(1) X(2)],[Y(yIn(i)) Y(yOut(j))],...
                     'Color',weightColor(w));
            end
        end
    end
    % nodes
    scatter(ax, X(1)*ones(1,numel(yIn)),  Y(yIn),    12,'w','filled');
    for h=1:L
        scatter(ax, X(h+1)*ones(1,numel(yH{h})), Y(yH{h}), 14,'w','filled');
    end
    scatter(ax, X(L+2)*ones(1,numel(yOut)), Y(yOut), 12,'w','filled');
    xlim(ax,[-0.5, X(L+2)+0.5]); ylim(ax,[-1.2 1.2]);
end


%% ───────────────────── Diversity Score ────────────────────────────────
function s = diversityScore(cre, baseW)
    Lcre = numel(cre.W)-1; Lbase = numel(baseW)-1;
    dL = abs(Lcre - Lbase);
    maxL = max(Lcre, Lbase);
    nCre = zeros(1,maxL); nBase = nCre;
    for k=1:Lcre,  nCre(k)  = size(cre.W{k},1);  end
    for k=1:Lbase, nBase(k) = size(baseW{k},1); end
    dN = sum(abs(nCre - nBase));
    dF = 0;
    for k=1:min(Lcre+1, Lbase+1)
        Wa = baseW{k}; Wc = cre.W{k};
        [rA,cA] = size(Wa); [rC,cC] = size(Wc);
        Wa(end+1:max(rA,rC), end+1:max(cA,cC)) = 0;
        Wc(end+1:max(rA,rC), end+1:max(cA,cC)) = 0;
        dF = dF + norm(Wa - Wc,'fro');
    end
    s = 1*dL + 0.2*dN + 0.002*dF;
end


%% ───────────────────── Toroidal Delta ─────────────────────────────────
function delta = torus(x,w)
    delta = mod(x + w/2, w) - w/2;
end

function y=clamp(x,a,b), y=max(a,min(b,x)); end
