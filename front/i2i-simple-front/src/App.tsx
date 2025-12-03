import styled from "styled-components";
import SimpleGraphCanvas from "./components/SimpleGraphCanvas";

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
  return (
    <AppContainer>
      <GraphContainer>
        <SimpleGraphCanvas />
      </GraphContainer>
    </AppContainer>
  );
}

export default App;
