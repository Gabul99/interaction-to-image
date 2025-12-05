import { useState, useMemo } from "react";
import styled from "styled-components";
import GraphCanvas from "./components/GraphCanvas";
import SimpleGraphCanvas from "./components/SimpleGraphCanvas";
import CompositionModal from "./components/CompositionModal";

const AppContainer = styled.div`
  height: 100vh;
  width: 100vw;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  position: relative;
  overflow: hidden;
  margin: 0;
  padding: 0;
`;

const GraphContainer = styled.div`
  width: 100%;
  height: 100vh;
  position: relative;
  overflow: hidden;
`;

function App() {
  const [compositionModalVisible, setCompositionModalVisible] = useState(false);

  // URL 쿼리 파라미터에서 mode와 participant 읽기
  const { mode, participant } = useMemo(() => {
    const params = new URLSearchParams(window.location.search);
    const modeParam = params.get("mode");
    const participantParam = params.get("p");
    // mode=prompt면 베이스라인, mode=step이거나 없으면 기존 시스템
    const modeValue = modeParam === "prompt" ? "prompt" : "step";
    const participantValue = participantParam ? parseInt(participantParam, 10) : null;
    return { mode: modeValue, participant: participantValue };
  }, []);

  const handleAddNodeClick = () => {
    setCompositionModalVisible(true);
  };

  const handleCompositionComplete = () => {
    setCompositionModalVisible(false);
    // 이미지 생성이 시작되면 GraphCanvas에서 처리됨
  };

  // mode에 따라 다른 컴포넌트 렌더링
  if (mode === "prompt") {
    // 베이스라인 모드 (SimpleGraphCanvas)
    return (
      <AppContainer>
        <GraphContainer>
          <SimpleGraphCanvas mode={mode} participant={participant} />
        </GraphContainer>
      </AppContainer>
    );
  }

  // 기본 모드 (기존 시스템 - GraphCanvas)
  return (
    <AppContainer>
      <GraphContainer>
        <GraphCanvas onAddNodeClick={handleAddNodeClick} mode={mode} participant={participant} />
      </GraphContainer>
      <CompositionModal
        visible={compositionModalVisible}
        onClose={() => setCompositionModalVisible(false)}
        onComplete={handleCompositionComplete}
      />
    </AppContainer>
  );
}

export default App;
