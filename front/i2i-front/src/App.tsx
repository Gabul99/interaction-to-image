import { useState } from "react";
import styled from "styled-components";
import GraphCanvas from "./components/GraphCanvas";
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

  const handleAddNodeClick = () => {
    setCompositionModalVisible(true);
  };

  const handleCompositionComplete = () => {
    setCompositionModalVisible(false);
    // 이미지 생성이 시작되면 GraphCanvas에서 처리됨
  };

  return (
    <AppContainer>
      <GraphContainer>
        <GraphCanvas onAddNodeClick={handleAddNodeClick} />
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
