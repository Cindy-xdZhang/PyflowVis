import pygame

class EventRegistrar:
    def __init__(self,imgui_impl):
        self.event_actions = []
        self.imgui_impl=imgui_impl
        self.running=True
      

    def register(self,  action):
        """   register a callback anction
        """
        self.event_actions.append(action)


    def handle_register_event(self,event):
        """
        handle all the customized events user registered in running time
        """
        for action in self.event_actions:
            action(event)
                    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                # Update the window size
                screen = pygame.display.set_mode((event.w, event.h), pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)
            
            self.handle_register_event(event)
            # Pass the pygame events to the ImGui Pygame renderer if we need imgui react(map pygame key to ImGui key etc.)
            self.imgui_impl.process_event(event) 
      
    def running(self)->bool:
        return self.running

if __name__ == '__main__':
    import unittest
    from unittest.mock import MagicMock, patch
    class TestEventRegistrar(unittest.TestCase):
        def setUp(self):
            # Mock imgui_impl before passing it to EventRegistrar
            self.imgui_impl = MagicMock()
            self.event_registrar = EventRegistrar(self.imgui_impl)

        def test_register(self):
            """Test if actions are correctly registered."""
            action = MagicMock()
            self.event_registrar.register(action)
            self.assertIn(action, self.event_registrar.event_actions)

        def test_handle_register_event(self):
            """Test if registered actions are called when handling events."""
            action = MagicMock()
            self.event_registrar.register(action)
            fake_event = MagicMock()
            self.event_registrar.handle_register_event(fake_event)
            action.assert_called_once_with(fake_event)

      

        @patch('pygame.event.get', return_value=[MagicMock()])
        def test_handle_events_custom(self, mock_event_get):
            """Test handling custom events."""
            action = MagicMock()
            self.event_registrar.register(action)
            self.event_registrar.handle_events()
            action.assert_called()
            self.imgui_impl.process_event.assert_called()
    unittest.main()