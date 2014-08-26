jQuery( document ).ready(function( $ ) {
    // Togglable side navigation.
    $( "#menuLink" ).click(function( event ) {
        event.preventDefault();
        $( "#layout" ).toggleClass( "active" );
        $( "#menu" ).toggleClass( "active" );
        $( "#menuLink" ).toggleClass( "active" );
    });
});
